"""Evaluation metrics for RDAE (per white paper §137)."""

from typing import Dict

import numpy as np
import torch


def compute_pose_error(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Compute end-effector pose error.

    Args:
        pred_pose: (..., 3) or (..., 7) predicted position [and orientation]
        gt_pose:   (..., 3) or (..., 7) ground-truth position [and orientation]
    Returns:
        scalar position error in meters
    """
    # Position error (Euclidean)
    pos_err = torch.norm(pred_pose[..., :3] - gt_pose[..., :3], dim=-1)

    # Orientation error (if quaternion provided)
    if pred_pose.shape[-1] >= 7:
        # Quaternion distance: min(‖q1 - q2‖, ‖q1 + q2‖)
        q1 = pred_pose[..., 3:7]
        q2 = gt_pose[..., 3:7]
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
        ori_err = torch.min(
            torch.norm(q1 - q2, dim=-1),
            torch.norm(q1 + q2, dim=-1),
        )
        return pos_err + 0.1 * ori_err

    return pos_err


def compute_joint_rmse(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
) -> torch.Tensor:
    """
    Compute RMSE for joint angle predictions.

    Args:
        pred_joints: (..., joint_dim)
        gt_joints:   (..., joint_dim)
    Returns:
        RMSE in radians
    """
    return torch.sqrt(torch.mean((pred_joints - gt_joints) ** 2, dim=-1))


def compute_force_error(
    pred_force: torch.Tensor,
    gt_force: torch.Tensor,
) -> torch.Tensor:
    """
    Compute force/torque prediction error (MAE).

    Args:
        pred_force: (..., 6)  [fx, fy, fz, tx, ty, tz]
        gt_force:   (..., 6)
    Returns:
        MAE in Newtons (force) / Nm (torque)
    """
    return torch.mean(torch.abs(pred_force - gt_force), dim=-1)


def compute_consistency_score(
    pose_error_cm: float,
    force_error_n: float,
    pose_threshold: float = 5.0,
    force_threshold: float = 0.5,
) -> float:
    """Combined consistency score in [0, 1]."""
    pose_score = max(0.0, 1.0 - pose_error_cm / pose_threshold)
    force_score = max(0.0, 1.0 - force_error_n / force_threshold)
    return float(0.5 * pose_score + 0.5 * force_score)
