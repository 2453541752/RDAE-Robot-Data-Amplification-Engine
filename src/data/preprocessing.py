"""Data preprocessing utilities."""

import torch
import torch.nn.functional as F
import torchvision.transforms as T


# Default image transforms
DEFAULT_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])


def preprocess_rgb(
    image: torch.Tensor,
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess a single RGB image.

    Args:
        image: (C, H, W) or (T, C, H, W), values in [0, 1] or [0, 255]
        target_size: (H, W)
        normalize: apply ImageNet normalization
    Returns:
        preprocessed tensor of same shape
    """
    if image.dim() == 4:
        # Time dimension
        return torch.stack([preprocess_rgb(img, target_size, normalize) for img in image])

    # Scale if needed
    if image.max() > 1.0:
        image = image / 255.0

    # Resize
    if image.shape[-2:] != target_size:
        image = F.interpolate(
            image.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

    # Normalize
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = (image - mean) / std

    return image


def preprocess_joint_state(
    joint_state: torch.Tensor,
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
) -> torch.Tensor:
    """
    Normalize joint state.

    Args:
        joint_state: (joint_dim,)
        mean, std: normalization statistics
    Returns:
        normalized joint state
    """
    if mean is not None and std is not None:
        joint_state = (joint_state - mean) / (std + 1e-8)
    return joint_state


def stack_frames(frames: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of frames into (T, C, H, W)."""
    return torch.stack(frames, dim=0)
