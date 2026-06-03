"""
RDAE Evaluation Script (MVP Stage 1)

Per white paper §137-145:
  - Compute action prediction error (MSE)
  - Compute joint RMSE and force MAE
  - Consistency score via simulation
  - Task success rate (requires downstream policy training)

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step0100000.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MultiModalWorldEncoder, ActionDecoder, SensorDecoder
from src.data import RobotDataset
from src.utils.config import load_config
from src.utils.metrics import compute_pose_error, compute_joint_rmse, compute_force_error, compute_consistency_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RDAE MVP Stage 1 Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="eval_results.json")
    return parser.parse_args()


def load_models(config: dict, checkpoint_path: str, device: torch.device):
    """Load models from checkpoint."""
    model_cfg = config["model"]

    encoder = MultiModalWorldEncoder(
        visual_backbone=model_cfg["visual_encoder"],
        visual_feature_dim=model_cfg["visual_feature_dim"],
        state_dim=model_cfg.get("state_dim", 14),
        world_latent_dim=model_cfg["world_latent_dim"],
        transformer_layers=model_cfg["transformer_layers"],
        transformer_heads=model_cfg["transformer_heads"],
    ).to(device)

    action_decoder = ActionDecoder(
        action_dim=model_cfg["action_dim"],
        action_horizon=model_cfg["action_horizon"],
        cond_dim=model_cfg["world_latent_dim"],
        diffusion_steps=model_cfg["diffusion_steps"],
    ).to(device)

    sensor_decoder = SensorDecoder(
        world_latent_dim=model_cfg["world_latent_dim"],
        action_dim=model_cfg["action_dim"],
        joint_dim=model_cfg["sensor_joint_dim"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    action_decoder.load_state_dict(ckpt["action_decoder"])
    sensor_decoder.load_state_dict(ckpt["sensor_decoder"])
    logger.info(f"Loaded checkpoint from step {ckpt['step']}")

    encoder.eval()
    action_decoder.eval()
    sensor_decoder.eval()

    return encoder, action_decoder, sensor_decoder


@torch.no_grad()
def evaluate(encoder, action_decoder, sensor_decoder, dataloader, device):
    """Run evaluation over the validation set."""
    metrics = {
        "action_mse": [],
        "joint_rmse": [],
        "force_mae": [],
        "consistency_score": [],
    }

    for batch in dataloader:
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        gt_actions = batch["actions"].to(device)

        # World latent
        z_w = encoder(images, state)

        # Generate actions
        pred_actions = action_decoder(z_w, None)

        # Action error
        if gt_actions.dim() == 2:
            gt_actions = gt_actions.permute(0, 2, 1)
        action_mse = torch.nn.functional.mse_loss(pred_actions, gt_actions)
        metrics["action_mse"].append(action_mse.item())

        # Sensor predictions
        sensor_preds = sensor_decoder(z_w, pred_actions)

        # Sensor errors
        if "joint_pos" in batch:
            joint_rmse = compute_joint_rmse(sensor_preds["joint_pos"], batch["joint_pos"].to(device))
            metrics["joint_rmse"].append(joint_rmse.mean().item())

        if "force" in batch:
            force_mae = compute_force_error(sensor_preds["force"], batch["force"].to(device))
            metrics["force_mae"].append(force_mae.mean().item())

        # Consistency score (simplified — full version requires simulation)
        # Here we compute a proxy based on prediction errors
        cs = compute_consistency_score(
            pose_error_cm=action_mse.item() * 10,  # rough conversion
            force_error_n=force_mae.mean().item() if "force" in batch else 0.0,
        )
        metrics["consistency_score"].append(cs)

    # Aggregate
    results = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, action_decoder, sensor_decoder = load_models(config, args.checkpoint, device)

    data_cfg = config["data"]
    dataset = RobotDataset(
        data_path=data_cfg["real_data_path"],
        dataset_type=data_cfg.get("real_dataset", "bridge_v2"),
        image_size=tuple(data_cfg["image_size"]),
        frame_stack=data_cfg["frame_stack"],
        action_horizon=config["model"]["action_horizon"],
        split="val",
    )
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                            num_workers=data_cfg["num_workers"])

    results = evaluate(encoder, action_decoder, sensor_decoder, dataloader, device)

    logger.info("=== Evaluation Results ===")
    for metric, stats in results.items():
        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save results
    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
