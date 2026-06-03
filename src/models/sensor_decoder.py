"""
Missing Sensor Reconstruction Module (MVP Stage 1)

Architecture (per white paper §71):
  - Inverse dynamics modeling: z_w + predicted actions → joint/force/tactile readings
  - 3-layer MLP for regression
  - Multi-task: joint angles, velocities, force/torque, tactile (if available)

Reference: AMPLIFY inverse model, LAPA
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class SensorDecoder(nn.Module):
    """
    Reconstructs robot internal sensor readings from world latent and actions.

    Predicts:
      - joint_positions: (B, joint_dim)      — joint angles (rad)
      - joint_velocities: (B, joint_dim)      — joint velocities (rad/s)
      - end_force: (B, 6)                     — wrench (3 force + 3 torque)
      - tactile: (B, tactile_dim)             — tactile array (optional)

    Architecture: world_latent ⊕ action → MLP → multi-head predictions
    """

    def __init__(
        self,
        world_latent_dim: int = 256,
        action_dim: int = 7,
        joint_dim: int = 7,
        tactile_dim: int = 0,                               # 0 = no tactile
        mlp_hidden: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.tactile_dim = tactile_dim

        input_dim = world_latent_dim + action_dim

        # Shared MLP trunk
        layers = []
        prev_dim = input_dim
        for h_dim in mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        trunk_out = mlp_hidden[-1]

        # Prediction heads
        self.joint_pos_head = nn.Linear(trunk_out, joint_dim)
        self.joint_vel_head = nn.Linear(trunk_out, joint_dim)
        self.force_head = nn.Linear(trunk_out, 6)            # 3 force + 3 torque

        # Optional tactile head
        if tactile_dim > 0:
            self.tactile_head = nn.Linear(trunk_out, tactile_dim)
        else:
            self.tactile_head = None

    def forward(
        self,
        z_w: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_w: world latent (B, world_latent_dim)
            actions: predicted action sequence (B, action_dim). If horizon > 1,
                     we use the first action step or mean-pool.
        Returns:
            Dict with keys: 'joint_pos', 'joint_vel', 'force', ('tactile')
        """
        # If actions have horizon dimension, pool to single vector
        if actions.dim() == 3:                               # (B, action_dim, horizon)
            actions = actions.mean(dim=-1)                    # (B, action_dim)

        x = torch.cat([z_w, actions], dim=-1)                 # (B, input_dim)
        h = self.trunk(x)                                     # (B, trunk_out)

        outputs = {
            "joint_pos": self.joint_pos_head(h),
            "joint_vel": self.joint_vel_head(h),
            "force": self.force_head(h),
        }

        if self.tactile_head is not None:
            outputs["tactile"] = self.tactile_head(h)

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task regression loss.

        Args:
            predictions: output from forward()
            targets: ground-truth sensor readings
            loss_weights: per-task loss weights, default equal
        Returns:
            total_loss, per_task_losses dict
        """
        if loss_weights is None:
            loss_weights = {}

        losses = {}
        total = torch.tensor(0.0, device=z_w.device if 'z_w' in dir() else predictions["joint_pos"].device)

        for key, pred in predictions.items():
            if key in targets:
                if key == "force":
                    # Split force/torque: L1 for robustness to outliers
                    loss = nn.functional.l1_loss(pred, targets[key])
                else:
                    loss = nn.functional.mse_loss(pred, targets[key])
                w = loss_weights.get(key, 1.0)
                losses[key] = loss
                total = total + w * loss

        return total, losses


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = SensorDecoder(
        world_latent_dim=256,
        action_dim=7,
        joint_dim=7,
        tactile_dim=0,
    ).to(device)

    # Dummy inputs
    z_w = torch.randn(4, 256).to(device)
    actions = torch.randn(4, 7, 16).to(device)              # with horizon

    preds = decoder(z_w, actions)
    for k, v in preds.items():
        print(f"  {k}: {v.shape}")

    # Loss computation
    targets = {
        "joint_pos": torch.randn(4, 7).to(device),
        "joint_vel": torch.randn(4, 7).to(device),
        "force": torch.randn(4, 6).to(device),
    }
    total_loss, per_loss = decoder.compute_loss(preds, targets)
    print(f"  total_loss={total_loss.item():.4f}")
    print(f"Params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")
