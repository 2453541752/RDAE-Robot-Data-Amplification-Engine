"""
Multi-Modal World Encoder (MVP Stage 1)

Architecture (per white paper §68):
  - Visual encoder: ResNet-50 or ViT-B/16 (pretrained), output 512-dim features
  - State encoder: small MLP for robot joint states
  - Fusion: 2-layer Transformer over concatenated [visual | state] tokens
  - Output: world latent z_w ∈ R^256

Reference: DINOv2, AMPLIFY, ViPRA
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ViTModel, ViTConfig


class VisualEncoder(nn.Module):
    """Visual backbone: ResNet-50 or ViT-B/16."""

    def __init__(
        self,
        backbone: str = "resnet50",
        output_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
            self.proj = nn.Linear(2048, output_dim)

        elif backbone == "vit_b_16":
            if pretrained:
                self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            else:
                config = ViTConfig(
                    image_size=224, patch_size=16, hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                )
                self.encoder = ViTModel(config)
            self.proj = nn.Linear(768, output_dim)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) or (B, T, C, H, W) stacked frames
        Returns:
            features: (B, output_dim) or (B, T, output_dim)
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            features = self._encode(images)
            return features.view(B, T, -1)
        return self._encode(images)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "resnet50":
            x = self.encoder(images)          # (B, 2048, 1, 1)
            x = x.flatten(1)                   # (B, 2048)
        else:
            x = self.encoder(images).pooler_output  # (B, 768)
        return self.proj(x)                    # (B, output_dim)


class StateEncoder(nn.Module):
    """Encodes robot proprioceptive state (joint angles, velocities, etc.)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            encoded: (B, output_dim) or (B, T, output_dim)
        """
        return self.net(state)


class MultiModalWorldEncoder(nn.Module):
    """
    World encoder that fuses visual and proprioceptive information into a
    compact world latent representation.

    Architecture:
      visual ──► VisualEncoder ──► [proj] ──┐
                                             ├──► Transformer ──► z_w (256-dim)
      state ───► StateEncoder ───► [proj] ──┘
    """

    def __init__(
        self,
        visual_backbone: str = "resnet50",
        visual_feature_dim: int = 512,
        state_dim: int = 14,                    # e.g., 7 joint angles + 7 velocities
        world_latent_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()

        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone, output_dim=visual_feature_dim
        )
        self.state_encoder = StateEncoder(
            state_dim=state_dim, output_dim=visual_feature_dim
        )

        # Learnable [CLS] token + positional embedding for Transformer fusion
        self.cls_token = nn.Parameter(torch.randn(1, 1, visual_feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_feature_dim,
            nhead=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Project to world latent space
        self.world_proj = nn.Linear(visual_feature_dim, world_latent_dim)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) or (B, T, C, H, W)
            state:  (B, state_dim) or (B, T, state_dim)
        Returns:
            z_w: (B, world_latent_dim) or (B, T, world_latent_dim)
        """
        has_time = images.dim() == 5
        B = images.shape[0]

        vis_feat = self.visual_encoder(images)    # (B, D) or (B, T, D)
        state_feat = self.state_encoder(state)     # (B, D) or (B, T, D)

        if has_time:
            T = vis_feat.shape[1]
            vis_feat = vis_feat.view(B * T, -1)
            state_feat = state_feat.view(B * T, -1)

        # Combine as token sequence: [CLS, visual, state]
        cls_tokens = self.cls_token.expand(vis_feat.shape[0], -1, -1)          # (N, 1, D)
        vis_tokens = vis_feat.unsqueeze(1)                                      # (N, 1, D)
        state_tokens = state_feat.unsqueeze(1)                                  # (N, 1, D)
        tokens = torch.cat([cls_tokens, vis_tokens, state_tokens], dim=1)       # (N, 3, D)

        # Transformer fusion
        fused = self.transformer(tokens)                                        # (N, 3, D)
        cls_out = fused[:, 0, :]                                                # (N, D)

        z_w = self.world_proj(cls_out)                                          # (N, world_latent_dim)

        if has_time:
            z_w = z_w.view(B, T, -1)

        return z_w


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = MultiModalWorldEncoder(
        visual_backbone="resnet50",
        state_dim=14,
        world_latent_dim=256,
    ).to(device)

    # Dummy batch
    batch_images = torch.randn(4, 3, 224, 224).to(device)
    batch_state = torch.randn(4, 14).to(device)

    z_w = encoder(batch_images, batch_state)
    print(f"Input:  images={batch_images.shape}, state={batch_state.shape}")
    print(f"Output: z_w={z_w.shape}")
    print(f"Params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")
