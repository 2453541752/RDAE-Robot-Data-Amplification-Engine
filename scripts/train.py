"""
RDAE Training Script (MVP Stage 1)

Per white paper §112-133:
  - Pre-training: self-supervised on video (contrastive, reconstruction)
  - Fine-tuning: supervised on real robot data with multi-task loss
  - Multi-task loss: λ1·action_mse + λ2·joint_mse + λ3·force_mae + λ4·consistency

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MultiModalWorldEncoder, ActionDecoder, SensorDecoder
from src.data import RobotDataset
from src.utils.config import load_config
from src.utils.metrics import compute_pose_error, compute_joint_rmse, compute_force_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RDAE MVP Stage 1 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true",
                        help="Run on small subset for debugging")
    return parser.parse_args()


def build_models(config: dict, device: torch.device):
    """Build encoder, action decoder, and sensor decoder."""
    model_cfg = config["model"]

    encoder = MultiModalWorldEncoder(
        visual_backbone=model_cfg["visual_encoder"],
        visual_feature_dim=model_cfg["visual_feature_dim"],
        state_dim=model_cfg.get("state_dim", 14),
        world_latent_dim=model_cfg["world_latent_dim"],
        transformer_layers=model_cfg["transformer_layers"],
        transformer_heads=model_cfg["transformer_heads"],
        transformer_dropout=model_cfg["transformer_dropout"],
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
        mlp_hidden=tuple(model_cfg["sensor_mlp_hidden"]),
    ).to(device)

    return encoder, action_decoder, sensor_decoder


def train_epoch(
    encoder, action_decoder, sensor_decoder,
    dataloader, optimizer, config, device, global_step, writer,
):
    """Single training epoch."""
    encoder.train()
    action_decoder.train()
    sensor_decoder.train()

    loss_weights = config["training"]["loss_weights"]

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        state = batch["state"].to(device)
        gt_actions = batch["actions"].to(device)

        # Forward: encode world latent
        z_w = encoder(images, state)

        # Action diffusion loss
        if gt_actions.dim() == 2:
            gt_actions = gt_actions.permute(0, 2, 1)   # (B, dim, horizon)
        noise_pred, noise = action_decoder(z_w, gt_actions)
        action_loss = nn.functional.mse_loss(noise_pred, noise)

        # Predict clean actions for sensor decoder input
        with torch.no_grad():
            pred_actions = action_decoder(z_w, None)    # (B, dim, horizon)

        # Sensor prediction loss
        sensor_preds = sensor_decoder(z_w, pred_actions)
        sensor_targets = {
            "joint_pos": batch.get("joint_pos", state[:, :7]).to(device),
            "joint_vel": batch.get("joint_vel", state[:, 7:14]).to(device),
            "force": batch.get("force", torch.zeros(state.shape[0], 6)).to(device),
        }
        sensor_loss, sensor_losses = sensor_decoder.compute_loss(
            sensor_preds, sensor_targets, loss_weights
        )

        # Total loss
        total_loss = (
            loss_weights["action_mse"] * action_loss +
            sensor_loss
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(action_decoder.parameters()) +
            list(sensor_decoder.parameters()),
            config["training"]["grad_clip_norm"],
        )
        optimizer.step()

        # Logging
        if global_step % config["training"]["log_interval"] == 0:
            logger.info(
                f"Step {global_step:7d} | "
                f"total={total_loss.item():.4f} | "
                f"action={action_loss.item():.4f} | "
                f"sensor={sensor_loss.item():.4f} | "
                f"grad={grad_norm:.2f}"
            )
            for name, loss_val in sensor_losses.items():
                writer.add_scalar(f"loss/{name}", loss_val.item(), global_step)
            writer.add_scalar("loss/total", total_loss.item(), global_step)
            writer.add_scalar("loss/action", action_loss.item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)

        global_step += 1

        # Validation
        if global_step % config["training"]["eval_interval"] == 0:
            # Placeholder for validation logic
            pass

        # Checkpoint
        if global_step % config["training"]["save_interval"] == 0:
            save_checkpoint(
                encoder, action_decoder, sensor_decoder,
                optimizer, global_step, config["training"].get("checkpoint_dir", "checkpoints"),
            )

    return global_step


def save_checkpoint(encoder, action_decoder, sensor_decoder, optimizer, step, ckpt_dir):
    """Save training checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step{step:07d}.pt")
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "action_decoder": action_decoder.state_dict(),
        "sensor_decoder": sensor_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(
        config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Build models
    encoder, action_decoder, sensor_decoder = build_models(config, device)
    total_params = sum(
        p.numel() for m in [encoder, action_decoder, sensor_decoder]
        for p in m.parameters()
    )
    logger.info(f"Total parameters: {total_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(action_decoder.parameters()) +
        list(sensor_decoder.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Dataset & DataLoader
    data_cfg = config["data"]
    dataset = RobotDataset(
        data_path=data_cfg["real_data_path"],
        dataset_type=data_cfg.get("real_dataset", "bridge_v2"),
        image_size=tuple(data_cfg["image_size"]),
        frame_stack=data_cfg["frame_stack"],
        action_horizon=config["model"]["action_horizon"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    logger.info(f"Dataset: {len(dataset)} episodes")

    # TensorBoard
    writer = SummaryWriter(log_dir=config["training"].get("log_dir", "logs"))

    # Training loop
    global_step = 0
    max_steps = config["training"]["max_steps"]
    logger.info(f"Starting training for {max_steps} steps...")

    while global_step < max_steps:
        global_step = train_epoch(
            encoder, action_decoder, sensor_decoder,
            dataloader, optimizer, config, device, global_step, writer,
        )

    logger.info("Training complete!")
    writer.close()


if __name__ == "__main__":
    main()
