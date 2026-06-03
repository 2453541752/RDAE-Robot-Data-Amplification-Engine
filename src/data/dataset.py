"""
Robot Dataset Loader

Supports:
  - BridgeData V2: HDF5-based, 60k trajectories, 24 environments
  - DROID: MCAP/ROS bag format, 76k trajectories, 564 scenes
  - Custom: generic dict-based interface

Data format (per white paper §77):
  Each sample includes: timestamp, RGB image(s), depth (optional),
  joint state, end-effector pose, action, force/torque, language (optional)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    """Unified dataset class for real robot demonstration data."""

    def __init__(
        self,
        data_path: str,
        dataset_type: str = "bridge_v2",
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        frame_stack: int = 2,
        action_horizon: int = 16,
        normalize: bool = True,
        transform=None,
    ):
        """
        Args:
            data_path: root directory of dataset
            dataset_type: "bridge_v2" | "droid" | "custom"
            split: "train" | "val" | "test"
            image_size: (H, W) to resize images
            frame_stack: number of consecutive frames to stack
            action_horizon: number of future actions to predict
            normalize: whether to normalize states/actions
            transform: optional image augmentation
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.split = split
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_horizon = action_horizon
        self.normalize = normalize
        self.transform = transform

        # Load episode index
        self.episodes = self._load_episodes()

        # Compute normalization stats if needed
        if self.normalize:
            self.stats = self._compute_or_load_stats()
        else:
            self.stats = {}

    def _load_episodes(self) -> List[Dict]:
        """Load episode metadata from disk."""
        if self.dataset_type == "bridge_v2":
            return self._load_bridge_v2()
        elif self.dataset_type == "droid":
            return self._load_droid()
        elif self.dataset_type == "custom":
            return self._load_custom()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _load_bridge_v2(self) -> List[Dict]:
        """
        BridgeData V2 format:
          - HDF5 files per trajectory
          - Keys: 'observations/images0', 'observations/state', 'actions', etc.
        """
        episodes = []
        # BridgeData V2 stores trajectories as .hdf5 files
        hdf5_files = sorted(self.data_path.glob("*.hdf5"))
        if not hdf5_files:
            hdf5_files = sorted(self.data_path.rglob("*.hdf5"))

        for h5_path in hdf5_files:
            try:
                import h5py
                with h5py.File(h5_path, "r") as f:
                    n_steps = f["actions"].shape[0]
                    episodes.append({
                        "path": str(h5_path),
                        "n_steps": n_steps,
                        "keys": list(f.keys()),
                    })
            except Exception as e:
                print(f"Warning: skipping {h5_path} — {e}")

        print(f"Loaded {len(episodes)} episodes from BridgeData V2 ({self.split})")
        return episodes

    def _load_droid(self) -> List[Dict]:
        """
        DROID format:
          - MCAP bag files
          - Each bag contains synchronized topics: /rgb, /joint_states, /action, etc.
        """
        episodes = []
        mcap_files = sorted(self.data_path.glob("*.mcap"))
        if not mcap_files:
            mcap_files = sorted(self.data_path.rglob("*.mcap"))

        for mcap_path in mcap_files:
            episodes.append({
                "path": str(mcap_path),
                "format": "mcap",
            })

        print(f"Loaded {len(episodes)} episodes from DROID ({self.split})")
        return episodes

    def _load_custom(self) -> List[Dict]:
        """Custom format: directory of .npz files or a manifest.json."""
        manifest = self.data_path / "manifest.json"
        if manifest.exists():
            import json
            with open(manifest) as f:
                return json.load(f)
        # Fallback: glob for .npz files
        npz_files = sorted(self.data_path.glob("*.npz"))
        return [{"path": str(p)} for p in npz_files]

    def _compute_or_load_stats(self) -> Dict[str, np.ndarray]:
        """Compute or load normalization statistics."""
        stats_path = self.data_path / f"stats_{self.split}.npz"
        if stats_path.exists():
            return dict(np.load(stats_path))

        # Placeholder: compute stats from first N episodes
        # In production, run a full pass over the dataset
        stats = {
            "state_mean": np.zeros(14, dtype=np.float32),
            "state_std": np.ones(14, dtype=np.float32),
            "action_mean": np.zeros(7, dtype=np.float32),
            "action_std": np.ones(7, dtype=np.float32),
        }
        return stats

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample:
          {
            "images": (C, H, W) or (frame_stack, C, H, W)
            "state": (state_dim,)
            "actions": (action_horizon, action_dim)
            "sensors": {"joint_pos": ..., "joint_vel": ..., "force": ...}
            "language": str (optional)
          }
        """
        ep = self.episodes[idx]

        if self.dataset_type == "bridge_v2":
            return self._get_bridge_item(ep)
        elif self.dataset_type == "droid":
            return self._get_droid_item(ep)
        else:
            return self._get_custom_item(ep)

    def _get_bridge_item(self, ep: Dict) -> Dict[str, torch.Tensor]:
        """Load a BridgeData V2 sample."""
        import h5py
        with h5py.File(ep["path"], "r") as f:
            # Sample a random start frame
            max_start = max(1, f["actions"].shape[0] - self.action_horizon - self.frame_stack)
            start = np.random.randint(0, max_start)

            # Load images
            images = []
            for i in range(self.frame_stack):
                img = f[f"observations/images0"][start + i]              # (H, W, C)
                img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            images = torch.stack(images) if self.frame_stack > 1 else images[0]  # (T, C, H, W)

            # Load state (joint angles + gripper)
            state = torch.from_numpy(
                f["observations/state"][start].astype(np.float32)
            )

            # Load action sequence
            actions = torch.from_numpy(
                f["actions"][start : start + self.action_horizon].astype(np.float32)
            )                                                                         # (horizon, action_dim)

            # Normalize
            if self.normalize:
                s_mean = torch.from_numpy(self.stats["state_mean"])
                s_std = torch.from_numpy(self.stats["state_std"])
                a_mean = torch.from_numpy(self.stats["action_mean"])
                a_std = torch.from_numpy(self.stats["action_std"])
                state = (state - s_mean) / (s_std + 1e-8)
                actions = (actions - a_mean) / (a_std + 1e-8)

            # Language instruction (if available)
            language = ""
            if "language" in f.attrs:
                language = f.attrs["language"]
            elif "language" in f:
                language = str(f["language"][()])

            return {
                "images": images,
                "state": state,
                "actions": actions,
                "language": language,
            }

    def _get_droid_item(self, ep: Dict) -> Dict[str, torch.Tensor]:
        """Load a DROID sample (placeholder — requires MCAP parsing)."""
        # TODO: Implement MCAP parsing with mcap + rosbags libraries
        raise NotImplementedError(
            "DROID MCAP parsing not yet implemented. "
            "Use BridgeData V2 or custom format for MVP."
        )

    def _get_custom_item(self, ep: Dict) -> Dict[str, torch.Tensor]:
        """Load a custom-format sample."""
        data = np.load(ep["path"], allow_pickle=True)
        return {
            "images": torch.from_numpy(data["images"]).float(),
            "state": torch.from_numpy(data["state"]).float(),
            "actions": torch.from_numpy(data["actions"]).float(),
            "language": str(data.get("language", "")),
        }


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("RobotDataset — awaiting real data at ./data/real/")
    print("Expected layout:")
    print("  data/real/bridge_v2/*.hdf5   — BridgeData V2")
    print("  data/real/droid/*.mcap        — DROID")
    print("  data/real/custom/*.npz        — Custom format")
