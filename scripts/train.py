"""
RDAE Training Script v2 (2026-06-03)

Updated for architecture v2:
  - MultiModalWorldEncoder: frozen ViT + Perceiver + 8-layer Causal Transformer
  - ActionDecoder: DDIM 10-step sampling
  - SensorDecoder: Temporal Transformer + contact detection + physics loss

Training phases:
  Phase A (pretrain): video prediction only (Ego4D)
  Phase B (finetune): multi-task on robot data (BridgeData V2 / DROID)
  Phase C (validate): simulation consistency check

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --phase pretrain
    python scripts/train.py --config configs/default.yaml --phase finetune --resume checkpoints/xxx.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MultiModalWorldEncoder, ActionDecoder, SensorDecoder
from src.data import RobotDataset
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RDAE v2 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--phase", type=str, default="finetune",
                        choices=["pretrain", "finetune", "validate"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Model building
# =============================================================================

def build_models(config: dict, device: torch.device):
    """Build encoder, action decoder, and sensor decoder."""
    m = config["model"]
    t = m["transformer"]
    p = m["perceiver"]
    s = m["sensor"]

    encoder = MultiModalWorldEncoder(
        vit_model=m["vit_model"],
        freeze_vit=m["freeze_vit"],
        perceiver_num_latents=p["num_latents"],
        perceiver_layers=p["num_layers"],
        transformer_dim=t["dim"],
        transformer_layers=t["num_layers"],
        transformer_heads=t["num_heads"],
        transformer_dropout=t["dropout"],
        state_dim=m["state_dim"],
    ).to(device)

    action_decoder = ActionDecoder(
        action_dim=m["action_dim"],
        action_horizon=m["action_horizon"],
        execution_horizon=m["execution_horizon"],
        cond_dim=m["world_latent_dim"],
        diffusion_steps=m["diffusion_steps"],
        ddim_steps=m["ddim_steps"],
        eta=m["ddim_eta"],
    ).to(device)

    sensor_decoder = SensorDecoder(
        world_latent_dim=m["world_latent_dim"],
        action_dim=m["action_dim"],
        inv_feat_dim=s["inv_feat_dim"],
        joint_dim=s["joint_dim"],
        temporal_layers=s["temporal_layers"],
        temporal_heads=s["temporal_heads"],
        temporal_hidden_dim=s["temporal_hidden"],
    ).to(device)

    return encoder, action_decoder, sensor_decoder


# =============================================================================
# Pretraining phase (video prediction)
# =============================================================================

def train_pretrain_epoch(
    encoder, dataloader, optimizer, config, device, global_step, writer, scaler,
):
    """Pretrain: learn world dynamics via video prediction."""
    encoder.train()
    loss_weight = config["training"]["loss_weights"].get("video_mse", 0.2)

    for batch in dataloader:
        images = batch["images"].to(device)                   # (B, T, C, H, W)
        state = batch["state"].to(device)                     # (B, T, 14)

        # Get current frame latents (first T-1 frames)
        current_images = images[:, :-1]                       # (B, T-1, C, H, W)
        current_state = state[:, :-1]                         # (B, T-1, 14)
        future_images = images[:, -1]                         # (B, C, H, W) — last frame

        with autocast(enabled=config["hardware"]["mixed_precision"] == "fp16"):
            outputs = encoder(current_images, current_state)

            # Get future frame ViT encoding for target
            with torch.no_grad():
                future_patches = encoder.visual_encoder(future_images)  # (B, 197, 768)
                future_target = future_patches.mean(dim=1)              # (B, 768)

            # Predict future frame latent from FRS token
            future_pred = outputs["future_frame_pred"]        # (B, dim)

            # For pretraining, align dimensions if needed
            if future_pred.shape[-1] != future_target.shape[-1]:
                future_target = future_target[:, :future_pred.shape[-1]]

            video_loss = nn.functional.mse_loss(future_pred, future_target)

        scaler.scale(video_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), config["training"]["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if global_step % config["training"]["log_interval"] == 0:
            logger.info(f"Step {global_step:7d} | video_loss={video_loss.item():.4f}")
            writer.add_scalar("pretrain/video_loss", video_loss.item(), global_step)

        global_step += 1
        if config.get("debug") and global_step >= 100:
            break

    return global_step


# =============================================================================
# Fine-tuning phase (multi-task on robot data)
# =============================================================================

def train_finetune_epoch(
    encoder, action_decoder, sensor_decoder,
    dataloader, optimizer, config, device, global_step, writer, scaler,
):
    """Finetune: multi-task learning with action + sensor losses."""
    encoder.train()
    action_decoder.train()
    sensor_decoder.train()

    lw = config["training"]["loss_weights"]

    for batch in dataloader:
        images = batch["images"].to(device)                   # (B, T, C, H, W)
        state = batch["state"].to(device)                     # (B, T, 14)
        gt_actions = batch["actions"].to(device)              # (B, horizon, action_dim) or (B, T, action_dim)

        with autocast(enabled=config["hardware"]["mixed_precision"] == "fp16"):
            # 1. World encoding
            encoder_outputs = encoder(images, state)
            z_w_global = encoder_outputs["z_w_global"]         # (B, 256)
            z_w_temporal = encoder_outputs["z_w_temporal"]     # (B, T, 256)
            inv_feat = encoder_outputs["inv_feat"]             # (B, 384)

            # 2. Action diffusion loss
            # For diffusion, we use the global latent and reshape actions
            if gt_actions.dim() == 3 and gt_actions.shape[1] == config["model"]["action_horizon"]:
                # (B, horizon, action_dim) → (B, action_dim, horizon)
                actions_for_diff = gt_actions.permute(0, 2, 1)
            else:
                # Use last T frames
                actions_for_diff = gt_actions[:, -1, :config["model"]["action_horizon"]]
                if actions_for_diff.dim() == 2:
                    actions_for_diff = actions_for_diff.unsqueeze(1).expand(-1, config["model"]["action_dim"], -1)

            noise_pred, noise = action_decoder(z_w_global, actions_for_diff)
            action_loss = nn.functional.mse_loss(noise_pred, noise)

            # Get generated actions for sensor decoder input
            with torch.no_grad():
                pred_actions = action_decoder.sample_ddim(z_w_global)

            # 3. Sensor prediction
            # Expand predicted actions to temporal dimension
            pred_actions_temporal = pred_actions.mean(dim=-1).unsqueeze(1).expand(
                -1, z_w_temporal.shape[1], -1
            )  # (B, T, action_dim) — simplified

            sensor_preds = sensor_decoder(z_w_temporal, pred_actions_temporal, inv_feat)

            # Build targets
            T = z_w_temporal.shape[1]
            sensor_targets = {
                "joint_pos": state[:, :, :config["model"]["sensor"]["joint_dim"]],
                "joint_vel": state[:, :, config["model"]["sensor"]["joint_dim"]:],
                "force": torch.zeros(images.shape[0], T, 6, device=device),
                "contact_mask": torch.zeros(images.shape[0], T, device=device),
            }
            # Use available targets from batch
            for key in ["joint_pos", "joint_vel", "force", "contact_mask"]:
                if key in batch:
                    sensor_targets[key] = batch[key].to(device)

            sensor_loss, sensor_losses = sensor_decoder.compute_loss(
                sensor_preds, sensor_targets, lw
            )

            # 4. Auxiliary future frame prediction loss
            if images.shape[1] >= 2:
                future_target = encoder.visual_encoder(images[:, -1])
                future_target_pooled = future_target.mean(dim=1)[:, :encoder_outputs["future_frame_pred"].shape[-1]]
                video_loss = nn.functional.mse_loss(
                    encoder_outputs["future_frame_pred"], future_target_pooled
                )
            else:
                video_loss = torch.tensor(0.0, device=device)

            # 5. Total loss
            total_loss = (
                lw["action_mse"] * action_loss +
                sensor_loss +
                lw["video_mse"] * video_loss
            )

        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(action_decoder.parameters()) +
            list(sensor_decoder.parameters()),
            config["training"]["grad_clip_norm"],
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Logging
        if global_step % config["training"]["log_interval"] == 0:
            log_msg = (
                f"Step {global_step:7d} | total={total_loss.item():.4f}"
                f" | action={action_loss.item():.4f}"
                f" | sensor={sensor_loss.item():.4f}"
                f" | video={video_loss.item():.4f}"
            )
            for name, loss_val in sensor_losses.items():
                log_msg += f" | {name}={loss_val.item():.4f}"
            logger.info(log_msg)

            writer.add_scalar("loss/total", total_loss.item(), global_step)
            writer.add_scalar("loss/action", action_loss.item(), global_step)
            writer.add_scalar("loss/sensor", sensor_loss.item(), global_step)
            writer.add_scalar("loss/video", video_loss.item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)

        global_step += 1

        # Checkpoint
        if global_step % config["training"]["save_interval"] == 0:
            save_checkpoint(
                encoder, action_decoder, sensor_decoder,
                optimizer, scaler, global_step,
                config["training"]["checkpoint_dir"],
            )

        if config.get("debug") and global_step >= 50:
            break

    return global_step


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(encoder, action_decoder, sensor_decoder, optimizer, scaler, step, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step:07d}.pt")
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "action_decoder": action_decoder.state_dict(),
        "sensor_decoder": sensor_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")


def load_checkpoint(path, encoder, action_decoder, sensor_decoder, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    action_decoder.load_state_dict(ckpt["action_decoder"])
    sensor_decoder.load_state_dict(ckpt["sensor_decoder"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    logger.info(f"Resumed from step {ckpt['step']}")
    return ckpt["step"]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(args.device or config["hardware"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info(f"Device: {device} | Phase: {args.phase}")

    # Build models
    encoder, action_decoder, sensor_decoder = build_models(config, device)

    total_params = sum(p.numel() for m in [encoder, action_decoder, sensor_decoder] for p in m.parameters())
    trainable_params = sum(p.numel() for m in [encoder, action_decoder, sensor_decoder] for p in m.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(
        [p for m in [encoder, action_decoder, sensor_decoder] for p in m.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = GradScaler(enabled=config["hardware"]["mixed_precision"] == "fp16")

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, encoder, action_decoder, sensor_decoder, optimizer, scaler, device)

    # DataLoader
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
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset)} episodes | Batches: {len(dataloader)}")

    # Training
    writer = SummaryWriter(log_dir=config["training"]["log_dir"])
    global_step = start_step
    max_steps = config["training"]["max_steps"]

    train_fn = train_pretrain_epoch if args.phase == "pretrain" else train_finetune_epoch
    logger.info(f"Training for {max_steps} steps ({args.phase})...")

    while global_step < max_steps:
        global_step = train_fn(
            encoder, action_decoder, sensor_decoder,
            dataloader, optimizer, config, device, global_step, writer, scaler,
        )

    # Final checkpoint
    save_checkpoint(encoder, action_decoder, sensor_decoder, optimizer, scaler, global_step,
                    config["training"]["checkpoint_dir"])
    logger.info("Training complete!")
    writer.close()


if __name__ == "__main__":
    main()
