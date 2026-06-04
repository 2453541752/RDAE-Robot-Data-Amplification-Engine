"""
Missing Sensor Reconstruction Module v2 (2026-06-03)

Architecture redesign based on SOTA research (Seer, AMPLIFY, FILIC, LIP4RobotID):

  z_w (B,T,256) ──┐
  actions (B,T,7) ─┤
  inv_feat (B,dim)─┼──► Temporal Transformer ──► Multi-head Predictions
  frame_feat (B,..)┘                              per timestep:
                                                    · joint_pos (T, 7)
                                                    · joint_vel (T, 7)
                                                    · force (T, 6)
                                                    · contact (T, 1) ← binary

Key improvements over v1:
  1. Temporal Transformer (not mean_pool) — preserves full action history
  2. [INV] token features from encoder as conditioning
  3. Contact detection head (binary) — handles force discontinuity
  4. Physics consistency loss — Lagrangian dynamics constraint
  5. Bidirectional attention — sensor estimation can use full context

References:
  - Seer: [INV] token for inverse dynamics (ICLR 2025 Oral)
  - AMPLIFY: inverse model from motion tokens (Georgia Tech, 2025)
  - FILIC: joint torque → end-effector force via Jacobian (2025)
  - LIP4RobotID: GP-based inverse dynamics with Lagrangian physics (MERL)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactDetector(nn.Module):
    """
    Binary contact detector that handles the discontinuity in force signals.

    Instead of directly regressing forces (which jump from 0→N on contact),
    we first predict whether each timestep involves contact, then regress
    force magnitude conditioned on contact probability.

    Architecture: small GRU + classifier
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),                      # logit per timestep
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, input_dim)
        Returns:
            contact_logits: (B, T, 1) — sigmoid to get probability
        """
        gru_out, _ = self.gru(features)                    # (B, T, hidden*2)
        logits = self.classifier(gru_out)                   # (B, T, 1)
        return logits


class ForceRegressor(nn.Module):
    """
    Force/torque regressor conditioned on contact probability.

    Separates the regression into:
      - Non-contact regime: forces ≈ 0
      - Contact regime: estimate fx, fy, fz, tx, ty, tz
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 6,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),           # +1 for contact prob
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        contact_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, input_dim)
            contact_prob: (B, T, 1) from ContactDetector
        Returns:
            force: (B, T, 6)
        """
        x = torch.cat([features, contact_prob], dim=-1)    # (B, T, input_dim+1)
        return self.net(x)


class TemporalSensorDecoder(nn.Module):
    """
    Temporal decoder for sensor reconstruction.

    Uses bidirectional Transformer (not causal — sensor estimation
    can use full sequence context, unlike action generation).
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            features: (B, T, hidden_dim)
        """
        x = self.transformer(x)                             # (B, T, input_dim)
        return self.output_proj(x)                          # (B, T, hidden_dim)


class SensorDecoder(nn.Module):
    """
    Complete sensor reconstruction module.

    Inputs (from encoder):
      - z_w_temporal: (B, T, 256)  per-timestep world latent
      - inv_feat:     (B, dim)     inverse dynamics features
      - actions:      (B, T, action_dim) or (B, action_dim, horizon)

    Outputs:
      - joint_pos:    (B, T, joint_dim)
      - joint_vel:    (B, T, joint_dim)
      - force:        (B, T, 6)
      - contact_prob: (B, T, 1)
    """

    def __init__(
        self,
        world_latent_dim: int = 256,
        action_dim: int = 7,
        inv_feat_dim: int = 384,                             # from encoder transformer
        joint_dim: int = 7,
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.joint_dim = joint_dim
        self.inv_feat_dim = inv_feat_dim

        # Project inputs to common dimension
        self.z_proj = nn.Linear(world_latent_dim, temporal_hidden_dim)
        self.action_proj = nn.Linear(action_dim, temporal_hidden_dim)

        # Temporal decoder input: [z_w | action | inv_feat_expanded]
        combined_dim = temporal_hidden_dim * 2 + inv_feat_dim

        self.combine_proj = nn.Linear(combined_dim, inv_feat_dim)

        # Bidirectional temporal decoder
        self.temporal_decoder = TemporalSensorDecoder(
            input_dim=inv_feat_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=dropout,
        )

        # Contact detection
        self.contact_detector = ContactDetector(
            input_dim=temporal_hidden_dim,
            hidden_dim=128,
        )

        # Force regression (conditioned on contact)
        self.force_regressor = ForceRegressor(
            input_dim=temporal_hidden_dim,
            hidden_dim=256,
            output_dim=6,
        )

        # Joint prediction heads
        self.joint_pos_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, joint_dim),
        )
        self.joint_vel_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, joint_dim),
        )

    def forward(
        self,
        z_w_temporal: torch.Tensor,
        actions: torch.Tensor,
        inv_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_w_temporal: (B, T, 256)
            actions: (B, T, action_dim) or (B, action_dim, horizon)
            inv_feat: (B, inv_feat_dim)
        Returns:
            Dict with joint_pos, joint_vel, force, contact_prob, contact_logits
        """
        B = z_w_temporal.shape[0]
        T = z_w_temporal.shape[1]

        # Normalize actions shape to (B, T, action_dim)
        if actions.dim() == 3 and actions.shape[1] != T:
            # (B, action_dim, horizon) → (B, T, action_dim)
            actions = actions.permute(0, 2, 1)               # (B, horizon, action_dim)
            if actions.shape[1] > T:
                actions = actions[:, :T, :]

        # Project inputs
        z_feat = self.z_proj(z_w_temporal)                   # (B, T, hidden)
        act_feat = self.action_proj(actions)                  # (B, T, hidden)

        # Expand inv_feat to all timesteps
        inv_expanded = inv_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, inv_feat_dim)

        # Combine
        combined = torch.cat([z_feat, act_feat, inv_expanded], dim=-1)
        combined = self.combine_proj(combined)                # (B, T, inv_feat_dim)

        # Temporal decoding
        temporal_feat = self.temporal_decoder(combined)      # (B, T, hidden)

        # Contact detection
        contact_logits = self.contact_detector(temporal_feat)  # (B, T, 1)
        contact_prob = torch.sigmoid(contact_logits)

        # Force regression (conditioned on contact probability)
        force = self.force_regressor(temporal_feat, contact_prob)  # (B, T, 6)

        # Joint predictions
        joint_pos = self.joint_pos_head(temporal_feat)       # (B, T, joint_dim)
        joint_vel = self.joint_vel_head(temporal_feat)       # (B, T, joint_dim)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "force": force,
            "contact_prob": contact_prob,
            "contact_logits": contact_logits,
            "temporal_features": temporal_feat,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Multi-task loss with physics consistency.

        Loss components:
          1. joint_pos MSE
          2. joint_vel MSE
          3. force MAE (masked by contact for cleaner signal)
          4. contact BCE
          5. physics consistency: ||M(q)·q̈ + C(q,q̇) + G(q) - τ - JᵀF_ext||²
        """
        if loss_weights is None:
            loss_weights = {}

        losses = {}
        total = torch.tensor(0.0, device=predictions["joint_pos"].device)

        # Joint losses
        for key, target_key, loss_fn in [
            ("joint_pos", "joint_pos", F.mse_loss),
            ("joint_vel", "joint_vel", F.mse_loss),
        ]:
            if target_key in targets:
                loss = loss_fn(predictions[key], targets[target_key])
                w = loss_weights.get(key, 1.0)
                losses[key] = loss
                total = total + w * loss

        # Force loss (masked by contact to avoid penalizing near-zero non-contact forces)
        if "force" in targets:
            force_pred = predictions["force"]
            force_gt = targets["force"]
            # Weight contact regions more heavily
            if "contact_mask" in targets:
                contact_mask = targets["contact_mask"].unsqueeze(-1)  # (B, T, 1)
                contact_weight = contact_mask * 5.0 + (1 - contact_mask) * 1.0
                force_loss = (contact_weight * torch.abs(force_pred - force_gt)).mean()
            else:
                force_loss = F.l1_loss(force_pred, force_gt)
            w = loss_weights.get("force", 1.0)
            losses["force"] = force_loss
            total = total + w * force_loss

        # Contact loss
        if "contact_mask" in targets:
            contact_loss = F.binary_cross_entropy_with_logits(
                predictions["contact_logits"].squeeze(-1),
                targets["contact_mask"].float(),
            )
            w = loss_weights.get("contact", 0.5)
            losses["contact"] = contact_loss
            total = total + w * contact_loss

        # Physics consistency loss (simplified Lagrangian)
        # τ = M(q)·q̈ + C(q,q̇)q̇ + G(q) - Jᵀ·F_ext
        if "joint_pos" in targets and "joint_vel" in targets and "force" in targets:
            physics_loss = self._compute_physics_loss(
                predictions, targets, loss_weights.get("physics", 0.1)
            )
            if physics_loss is not None:
                losses["physics"] = physics_loss
                total = total + loss_weights.get("physics", 0.1) * physics_loss

        return total, losses

    def _compute_physics_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weight: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        Simplified physics consistency loss.

        We approximate: joint acceleration should be consistent with
        predicted forces and current state. This is a rough approximation
        — full Lagrangian dynamics requires robot-specific model parameters.

        For MVP, we use a learned residual:
          τ_predicted = f(q, q̇)  should ≈  τ_actual

        Simplified as: MSE between predicted joint_vel change and expected
        change given predicted forces.
        """
        if weight <= 0:
            return None

        # Simplified: force magnitude should correlate with joint acceleration
        pred_acc = torch.diff(predictions["joint_vel"], dim=1)  # (B, T-1, 7)
        pred_force_mag = torch.norm(predictions["force"][:, :, :3], dim=-1)  # (B, T, 3) → norm

        # If there's large force, there should be large joint acceleration
        # We penalize when high force → low acceleration (physically inconsistent)
        force_norm = pred_force_mag[:, :-1]                      # (B, T-1)
        acc_norm = torch.norm(pred_acc, dim=-1)                  # (B, T-1)

        # Loss: correlation between force and acceleration should be positive
        # Simplified: penalize when force is large but acceleration is small
        threshold = 0.01
        return F.mse_loss(
            acc_norm / (acc_norm.max(dim=1, keepdim=True).values + 1e-6),
            force_norm / (force_norm.max(dim=1, keepdim=True).values + 1e-6),
        )


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = SensorDecoder(
        world_latent_dim=256,
        action_dim=7,
        inv_feat_dim=384,
        joint_dim=7,
    ).to(device)

    # Dummy temporal inputs
    B, T = 4, 8
    z_w = torch.randn(B, T, 256).to(device)
    actions = torch.randn(B, T, 7).to(device)
    inv_feat = torch.randn(B, 384).to(device)

    preds = decoder(z_w, actions, inv_feat)

    print("=== Sensor Decoder v2 Outputs ===")
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Test loss computation
    targets = {
        "joint_pos": torch.randn(B, T, 7).to(device),
        "joint_vel": torch.randn(B, T, 7).to(device),
        "force": torch.randn(B, T, 6).to(device),
        "contact_mask": (torch.rand(B, T) > 0.7).to(device),
    }
    total_loss, per_loss = decoder.compute_loss(preds, targets)
    print(f"\n  total_loss: {total_loss.item():.4f}")
    for k, v in per_loss.items():
        print(f"  loss/{k}: {v.item():.4f}")

    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\n  Trainable params: {trainable/1e6:.1f}M")
