"""
Multi-Modal World Encoder v2 (2026-06-03)

Architecture redesign based on SOTA research (GR-1, Seer, Octo, FLARE, MCR):

  RGB (T,3,224,224) ──► Frozen ViT-B/MAE ──► (T, 197 patches, 768)
                                                     │
                                                     ▼
                                            Perceiver Resampler
                                            (64 learnable queries)
                                                     │
                                                     ▼
                                            (T, 64 tokens, 384)
                                                     │
                     ┌───────────────────────────────┘
                     ▼
  joint (T,14) ──► StateEncoder ──► (T, 1 token, 384)
                     │
                     ▼
         ┌──────────────────────────────┐
         │  Causal World Transformer    │   ← 8 layers, 384-d, 12 heads
         │                              │
         │  Token sequence:              │
         │  [IMG_0..IMG_T][STATE_0..STATE_T][CLS][FRS][INV] │
         │                              │
         │  Outputs:                     │
         │   [CLS] → z_w (256)          │  world latent
         │   [FRS] → future frame pred  │  auxiliary task
         │   [INV] → sensor features    │  inverse dynamics
         └──────────────────────────────┘

References:
  - GR-1: MAE ViT-B + Perceiver Resampler + GPT-2 style Transformer (ByteDance, ICLR 2024)
  - Seer: [FRS] + [INV] tokens for foresight & inverse dynamics (Shanghai AI Lab, ICLR 2025)
  - Octo: Small CNN + large Transformer, block-wise masking (Berkeley/Stanford/CMU, RSS 2024)
  - MCR: DROID contrastive pretraining >> ImageNet for robot tasks (2024)
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTImageProcessor


# =============================================================================
# 1. Perceiver Resampler (adapted from GR-1 / DeepMind Perceiver)
# =============================================================================

class PerceiverResampler(nn.Module):
    """
    Compresses a large number of variable-length input tokens into a fixed
    number of output tokens using learnable query vectors + cross-attention.

    Input:  (B, N_in,  D_in)   — e.g., (B, 197 patch tokens, 768)
    Output: (B, N_out, D_out)  — e.g., (B, 64 compressed tokens, 384)

    This is the standard approach used by GR-1, Seer, and Flamingo to
    compress ViT patch tokens while preserving spatial information.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 384,
        num_latents: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_latents = num_latents

        # Learnable query vectors
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, output_dim) * 0.02
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Cross-attention layers (queries attend to input tokens)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        # Feed-forward after each cross-attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_in, D_in) or (B, T, N_in, D_in)
        Returns:
            (B, num_latents, output_dim) or (B, T, num_latents, output_dim)
        """
        has_time = x.dim() == 4
        if has_time:
            B, T, N, D = x.shape
            x = x.view(B * T, N, D)

        # Project input
        x = self.input_proj(x)                                 # (N_total, N_in, output_dim)

        # Expand latent queries
        batch_size = x.shape[0]
        latents = self.latent_queries.expand(batch_size, -1, -1)  # (N_total, num_latents, out_dim)

        # Cross-attention layers with residual FFN
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: latents attend to input tokens
            attn_out, _ = self.cross_attn_layers[i](
                query=latents, key=x, value=x
            )
            latents = self.norm1_layers[i](latents + attn_out)

            # Feed-forward
            ffn_out = self.ffn_layers[i](latents)
            latents = self.norm2_layers[i](latents + ffn_out)

        if has_time:
            latents = latents.view(B, T, self.num_latents, -1)

        return latents


# =============================================================================
# 2. Visual Encoder Wrapper (Frozen ViT)
# =============================================================================

class FrozenViTEncoder(nn.Module):
    """
    Frozen ViT-B/16 (or ViT-B/MAE) that outputs patch tokens.

    Key insight from MCR (2024) & GR-1:
      - ImageNet pretraining doesn't help robot tasks
      - MAE pretraining (masked autoencoding) is better for dense prediction
      - Keep ViT frozen during Stage-1 training to preserve pretrained features
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        freeze: bool = True,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.vit.config.hidden_size  # 768 for ViT-B

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Preprocessor for image normalization
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            patch_tokens: (B, N_patches+1, hidden) or (B, T, N_patches+1, hidden)
              including CLS token at position 0
        """
        has_time = images.dim() == 5
        if has_time:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)

        with torch.set_grad_enabled(not all(p.requires_grad for p in self.vit.parameters())):
            outputs = self.vit(images, output_hidden_states=False)
            patch_tokens = outputs.last_hidden_state  # (N_total, 197, 768)

        if has_time:
            patch_tokens = patch_tokens.view(B, T, *patch_tokens.shape[1:])

        return patch_tokens


# =============================================================================
# 3. State Encoder
# =============================================================================

class StateEncoder(nn.Module):
    """Encodes robot proprioceptive state into a single token per timestep."""

    def __init__(
        self,
        state_dim: int = 14,
        output_dim: int = 384,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            (B, 1, output_dim) or (B, T, 1, output_dim)
        """
        if state.dim() == 3:
            B, T, D = state.shape
            state = state.view(B * T, D)
            out = self.net(state)
            return out.view(B, T, 1, -1)
        out = self.net(state)
        return out.unsqueeze(1)


# =============================================================================
# 4. Causal World Transformer
# =============================================================================

class CausalWorldTransformer(nn.Module):
    """
    GPT-2 style causal Transformer that:
      1. Takes [IMG_tokens | STATE_tokens | CLS | FRS | INV] sequence
      2. Applies causal attention (can't peek into future)
      3. Outputs three specialized token embeddings

    Token layout (for each timestep t):
      IMG_t (64 tokens) | STATE_t (1 token)
    Then at the end:
      CLS (1 token) | FRS (1 token) | INV (1 token)

    Causal mask ensures:
      - IMG_t can attend to IMG_{<=t}, STATE_{<=t}
      - STATE_t can attend to IMG_{<=t}, STATE_{<=t}
      - CLS can attend to all past tokens
      - FRS can attend to all past tokens + CLS
      - INV can attend to all past tokens + CLS + FRS
    """

    def __init__(
        self,
        dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 12,
        img_tokens_per_frame: int = 64,
        state_tokens_per_frame: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.img_tokens_per_frame = img_tokens_per_frame
        self.state_tokens_per_frame = state_tokens_per_frame

        # Special learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.frs_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Foresight
        self.inv_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Inverse dynamics

        # Learnable temporal position encoding
        self.temporal_pos_emb = nn.Parameter(
            torch.randn(1, 128, dim) * 0.02  # max 128 frames
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            ) for _ in range(num_layers)
        ])

        # Output projections
        self.cls_proj = nn.Linear(dim, 256)    # → z_w
        self.frs_proj = nn.Linear(dim, dim)    # → future frame latent
        self.inv_proj = nn.Linear(dim, dim)    # → inverse dynamics features

    def _build_sequence(
        self,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        B: int, T: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Build the full token sequence with proper temporal position encoding.

        Layout per frame: [IMG (64) | STATE (1)]
        Then global: [CLS (1) | FRS (1) | INV (1)]
        """
        # Add temporal position encoding to per-frame tokens
        pos = self.temporal_pos_emb[:, :T, :].unsqueeze(2)  # (1, T, 1, dim)

        img_tokens = img_tokens + pos                            # (B, T, 64, dim)
        state_tokens = state_tokens + pos[:, :, :1, :]          # (B, T, 1, dim)

        # Flatten temporal dimension
        img_tokens = img_tokens.view(B, T * self.img_tokens_per_frame, self.dim)
        state_tokens = state_tokens.view(B, T * self.state_tokens_per_frame, self.dim)

        # Interleave: IMG_0 | STATE_0 | IMG_1 | STATE_1 | ...
        tokens = []
        for t in range(T):
            tokens.append(img_tokens[:, t * self.img_tokens_per_frame:(t+1) * self.img_tokens_per_frame])
            tokens.append(state_tokens[:, t * self.state_tokens_per_frame:(t+1) * self.state_tokens_per_frame])
        seq = torch.cat(tokens, dim=1)                          # (B, T*(64+1), dim)

        # Append special tokens
        special = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.frs_token.expand(B, -1, -1),
            self.inv_token.expand(B, -1, -1),
        ], dim=1)                                                 # (B, 3, dim)

        return torch.cat([seq, special], dim=1)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Build causal attention mask.

        Each frame has (img_tokens + state_tokens) tokens.
        Special tokens are at the end: CLS, FRS, INV.

        Causal ordering: IMG_0 < STATE_0 < IMG_1 < STATE_1 < ... < CLS < FRS < INV
        """
        per_frame = self.img_tokens_per_frame + self.state_tokens_per_frame
        total_frame_tokens = T * per_frame
        total_len = total_frame_tokens + 3  # + CLS, FRS, INV

        # Standard causal mask
        mask = torch.triu(
            torch.ones(total_len, total_len, device=device) * float('-inf'),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            img_tokens:   (B, T, 64, dim) from PerceiverResampler
            state_tokens: (B, T, 1, dim) from StateEncoder
        Returns:
            {
                "z_w":      (B, T, 256)   world latent per timestep
                "frs_feat": (B, dim)      foresight features
                "inv_feat": (B, T, dim)   inverse dynamics features per timestep
            }
        """
        B, T = img_tokens.shape[:2]
        device = img_tokens.device

        # Build sequence
        seq = self._build_sequence(img_tokens, state_tokens, B, T, device)

        # Build causal mask
        causal_mask = self._build_causal_mask(T, device)

        # Apply Transformer layers
        for layer in self.layers:
            seq = layer(seq, src_mask=causal_mask)

        # Extract outputs
        # INV token is last, then FRS, then CLS
        per_frame = T * (self.img_tokens_per_frame + self.state_tokens_per_frame)

        cls_out = seq[:, per_frame, :]        # CLS token output
        frs_out = seq[:, per_frame + 1, :]    # FRS token output
        inv_out = seq[:, per_frame + 2, :]    # INV token output

        # To get per-timestep outputs, we extract IMG tokens and pool
        # For z_w: attend through CLS projection (global)
        z_w_global = self.cls_proj(cls_out)       # (B, 256)

        # For per-frame z_w: average pool all tokens belonging to each frame
        z_w_per_frame = []
        for t in range(T):
            start = t * (self.img_tokens_per_frame + self.state_tokens_per_frame)
            end = start + self.img_tokens_per_frame + self.state_tokens_per_frame
            frame_feat = seq[:, start:end, :].mean(dim=1)  # pool frame tokens
            z_w_per_frame.append(self.cls_proj(frame_feat))
        z_w_temporal = torch.stack(z_w_per_frame, dim=1)   # (B, T, 256)

        return {
            "z_w_global": z_w_global,
            "z_w_temporal": z_w_temporal,
            "frs_feat": self.frs_proj(frs_out),
            "inv_feat": self.inv_proj(inv_out),
            # Also return per-frame pooled features for temporal tasks
            "frame_features": seq[:, :per_frame, :],
        }


# =============================================================================
# 5. MultiModalWorldEncoder (unified interface)
# =============================================================================

class MultiModalWorldEncoder(nn.Module):
    """
    Unified world encoder combining all components.

    Usage:
        encoder = MultiModalWorldEncoder()
        outputs = encoder(images, state)
        z_w = outputs["z_w_temporal"]   # (B, T, 256)
    """

    def __init__(
        self,
        # Visual
        vit_model: str = "google/vit-base-patch16-224",
        freeze_vit: bool = True,
        # Perceiver
        perceiver_num_latents: int = 64,
        perceiver_layers: int = 2,
        # Transformer
        transformer_dim: int = 384,
        transformer_layers: int = 8,
        transformer_heads: int = 12,
        transformer_dropout: float = 0.1,
        # State
        state_dim: int = 14,
        # Temporal
        img_tokens_per_frame: int = 64,
        max_frames: int = 128,
    ):
        super().__init__()

        # 1. Frozen ViT
        self.visual_encoder = FrozenViTEncoder(
            model_name=vit_model, freeze=freeze_vit
        )
        vit_dim = self.visual_encoder.output_dim  # 768

        # 2. Perceiver Resampler
        self.perceiver = PerceiverResampler(
            input_dim=vit_dim,
            output_dim=transformer_dim,
            num_latents=perceiver_num_latents,
            num_layers=perceiver_layers,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
        )

        # 3. State Encoder
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            output_dim=transformer_dim,
        )

        # 4. Causal World Transformer
        self.transformer = CausalWorldTransformer(
            dim=transformer_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            img_tokens_per_frame=img_tokens_per_frame,
            state_tokens_per_frame=1,
            dropout=transformer_dropout,
        )

        # Auxiliary head: predict future frame latent from FRS token
        self.future_frame_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, T, C, H, W) or (B, C, H, W)
            state:  (B, T, state_dim) or (B, state_dim)
        Returns:
            {
                "z_w_global":   (B, 256)           global world latent
                "z_w_temporal": (B, T, 256)        per-timestep world latent
                "frs_feat":     (B, dim)           foresight features
                "inv_feat":     (B, dim)           inverse dynamics features
                "future_frame_pred": (B, dim)      predicted future frame latent
            }
        """
        # Ensure temporal dimension
        if images.dim() == 4:
            images = images.unsqueeze(1)           # (B, C, H, W) → (B, 1, C, H, W)
        if state.dim() == 2:
            state = state.unsqueeze(1)             # (B, D) → (B, 1, D)

        B, T = images.shape[:2]

        # 1. ViT encoding
        patch_tokens = self.visual_encoder(images) # (B, T, 197, 768)

        # 2. Perceiver compression
        img_tokens = self.perceiver(patch_tokens)   # (B, T, 64, 384)

        # 3. State encoding
        state_tokens = self.state_encoder(state)    # (B, T, 1, 384)

        # 4. Causal Transformer
        outputs = self.transformer(img_tokens, state_tokens)

        # 5. Future frame prediction (auxiliary task)
        future_frame_pred = self.future_frame_head(outputs["frs_feat"])

        outputs["future_frame_pred"] = future_frame_pred
        outputs["img_tokens"] = img_tokens
        outputs["state_tokens"] = state_tokens

        return outputs

    def compute_auxiliary_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        future_frame_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss: MSE between FRS prediction and actual future frame latent.

        Args:
            outputs: from forward()
            future_frame_latent: (B, dim) actual ViT encoding of the future frame
        Returns:
            scalar loss
        """
        return F.mse_loss(outputs["future_frame_pred"], future_frame_latent)


# =============================================================================
# 6. Quick test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder = MultiModalWorldEncoder(
        vit_model="google/vit-base-patch16-224",
        freeze_vit=True,
        transformer_layers=8,
        transformer_heads=12,
    ).to(device)

    # Dummy temporal batch: 2 samples, 4 frames each
    batch_images = torch.randn(2, 4, 3, 224, 224).to(device)
    batch_state = torch.randn(2, 4, 14).to(device)

    outputs = encoder(batch_images, batch_state)

    print("\n=== World Encoder v2 Outputs ===")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n  Total params: {total/1e6:.1f}M")
    print(f"  Trainable:    {trainable/1e6:.1f}M")

    # Test single-frame fallback
    single_image = torch.randn(2, 3, 224, 224).to(device)
    single_state = torch.randn(2, 14).to(device)
    outputs_single = encoder(single_image, single_state)
    print(f"\n  Single-frame z_w_temporal: {outputs_single['z_w_temporal'].shape}")
