"""
Action / Trajectory Decoder (MVP Stage 1)

Architecture (per white paper §72):
  - Diffusion Policy (π₀-style) for high-frequency continuous action generation
  - 100-step denoising diffusion
  - Predicts 16-step action horizon at ~22Hz control frequency
  - Conditioned on world latent z_w and optional language instruction

Reference: Diffusion Policy (Chi et al.), π₀ (Physical Intelligence)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditionalUNet1D(nn.Module):
    """
    1D U-Net conditioned on world latent for action denoising.
    Operates over the action horizon dimension as the "spatial" axis.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Condition projection (world latent → conditioning)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder (down-sampling over horizon)
        self.enc1 = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )

        # Decoder (up-sampling over horizon)
        self.dec1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(hidden_dim, action_dim, kernel_size=5, padding=2),
        )

        # FiLM modulation: time + condition → scale/shift for each layer
        self.film_layers = nn.ModuleList([
            nn.Linear(time_dim + hidden_dim, hidden_dim * 2) for _ in range(7)
        ])

    def _film(self, x: torch.Tensor, idx: int, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning."""
        film_in = torch.cat([t_emb, c_emb], dim=-1)            # (B, time_dim + hidden_dim)
        scale_shift = self.film_layers[idx](film_in)            # (B, hidden_dim*2)
        scale, shift = scale_shift.chunk(2, dim=-1)             # (B, hidden_dim) each
        return x * scale.unsqueeze(-1) + shift.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: noisy actions (B, action_dim, action_horizon)
            t: diffusion timestep (B,)
            cond: world latent z_w (B, cond_dim)
        Returns:
            predicted noise (B, action_dim, action_horizon)
        """
        t_emb = self.time_mlp(t)                                # (B, time_dim)
        c_emb = self.cond_proj(cond)                            # (B, hidden_dim)

        # Encoder
        e1 = self.enc1(x)                                       # (B, hidden, T)
        e1 = self._film(e1, 0, t_emb, c_emb)

        e2 = self.enc2(e1)
        e2 = self._film(e2, 1, t_emb, c_emb)

        e3 = self.enc3(e2)
        e3 = self._film(e3, 2, t_emb, c_emb)

        # Bottleneck
        b = self.bottleneck(e3)
        b = self._film(b, 3, t_emb, c_emb)

        # Decoder with skip connections
        d1 = self.dec1(b + e3)
        d1 = self._film(d1, 4, t_emb, c_emb)

        d2 = self.dec2(d1 + e2)
        d2 = self._film(d2, 5, t_emb, c_emb)

        d3 = self.dec3(d2 + e1)
        d3 = self._film(d3, 6, t_emb, c_emb)

        return d3


class ActionDecoder(nn.Module):
    """
    Diffusion-based action decoder.

    Given world latent z_w, generates action sequences through iterative denoising.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        cond_dim: int = 256,
        diffusion_steps: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.diffusion_steps = diffusion_steps

        self.unet = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
        )

        # Noise schedule (cosine)
        self.register_buffer(
            "beta",
            self._cosine_beta_schedule(diffusion_steps),
        )
        alpha = 1.0 - self.beta
        self.register_buffer("alpha_cumprod", torch.cumprod(alpha, dim=0))
        self.register_buffer("alpha_cumprod_prev", F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0))

    @staticmethod
    def _cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from improved DDPM."""
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) ** 2 + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, max=0.999)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean actions."""
        noise = torch.randn_like(x0)
        sqrt_alpha = self.alpha_cumprod[t].sqrt()
        sqrt_one_minus_alpha = (1 - self.alpha_cumprod[t]).sqrt()
        # Broadcast over spatial dims
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def forward(
        self,
        z_w: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training: predict noise from noisy actions.
        Inference: denoise from pure noise to generate actions.

        Args:
            z_w: world latent (B, cond_dim)
            actions: ground-truth action sequence (B, action_dim, action_horizon), None for inference
        Returns:
            predicted clean actions (B, action_dim, action_horizon)
        """
        if actions is not None:
            # Training: add noise and predict
            B = actions.shape[0]
            t = torch.randint(0, self.diffusion_steps, (B,), device=actions.device)
            xt, noise = self.add_noise(actions, t)
            noise_pred = self.unet(xt, t, z_w)
            return noise_pred, noise
        else:
            # Inference: iterative denoising (DDPM sampling)
            return self._sample(z_w)

    @torch.no_grad()
    def _sample(self, z_w: torch.Tensor) -> torch.Tensor:
        """DDPM sampling from pure noise."""
        B = z_w.shape[0]
        device = z_w.device

        x = torch.randn(B, self.action_dim, self.action_horizon, device=device)

        for i in reversed(range(self.diffusion_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t, z_w)

            alpha = 1.0 - self.beta[i]
            alpha_cumprod = self.alpha_cumprod[i]
            alpha_cumprod_prev = self.alpha_cumprod_prev[i]
            beta = self.beta[i]

            # DDPM posterior mean
            coef1 = beta / (1 - alpha_cumprod).sqrt()
            coef2 = (1 - alpha_cumprod_prev) * alpha.sqrt() / (1 - alpha_cumprod)
            x0_pred = (x - coef1 * noise_pred) / alpha_cumprod.sqrt()
            x = alpha_cumprod_prev.sqrt() * x0_pred + (1 - alpha_cumprod_prev).sqrt() * noise_pred

            # Add noise for intermediate steps
            if i > 0:
                noise = torch.randn_like(x)
                x = x + beta.sqrt() * noise

        return x


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = ActionDecoder(
        action_dim=7,
        action_horizon=16,
        cond_dim=256,
        diffusion_steps=100,
    ).to(device)

    # Dummy world latent
    z_w = torch.randn(2, 256).to(device)

    # Training forward
    gt_actions = torch.randn(2, 7, 16).to(device)
    noise_pred, noise = decoder(z_w, gt_actions)
    print(f"Training: noise_pred={noise_pred.shape}, noise={noise.shape}")

    # Inference sampling
    actions = decoder(z_w, None)
    print(f"Inference: generated actions={actions.shape}")
    print(f"Params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")
