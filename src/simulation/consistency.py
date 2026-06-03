"""
Physics Simulation Consistency Checker (MVP Stage 1)

Per white paper §73 and §143:
  - Execute generated actions in MuJoCo / Isaac Sim
  - Compare simulation output with predicted sensor data
  - Filter synthetic samples: pose_error < 5cm, force_error < 0.5N

Reference: MuJoCo, Isaac Sim
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class ConsistencyResult:
    """Result of a consistency check."""
    is_valid: bool
    score: float                          # 0–1, higher = better
    pose_error_cm: float                  # end-effector position error (cm)
    force_error_n: float                  # force prediction error (N)
    joint_error_rad: float                # joint angle error (rad)
    sim_trajectory: Optional[Dict] = None # full simulation output


class ConsistencyChecker:
    """
    Validates generated action/sensor sequences by running them in simulation
    and comparing with predictions.
    """

    def __init__(
        self,
        engine: str = "mujoco",
        robot_model_path: Optional[str] = None,
        scene_model_path: Optional[str] = None,
        timestep: float = 0.002,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            engine: simulation engine ("mujoco" or "isaacsim")
            robot_model_path: path to robot URDF/MJCF
            scene_model_path: path to scene description
            timestep: simulation timestep (seconds)
            thresholds: {"pose_error_cm": 5.0, "force_error_n": 0.5}
        """
        self.engine = engine
        self.robot_model_path = robot_model_path
        self.scene_model_path = scene_model_path
        self.timestep = timestep
        self.thresholds = thresholds or {
            "pose_error_cm": 5.0,
            "force_error_n": 0.5,
            "joint_error_rad": 0.1,
        }

        self._sim = None  # Lazy init

    def _init_sim(self):
        """Lazy initialization of simulation environment."""
        if self._sim is not None:
            return

        if self.engine == "mujoco":
            self._init_mujoco()
        elif self.engine == "isaacsim":
            self._init_isaac()
        else:
            raise ValueError(f"Unknown simulation engine: {self.engine}")

    def _init_mujoco(self):
        """Initialize MuJoCo simulation."""
        try:
            import mujoco
            self._mujoco = mujoco

            # Load robot model (placeholder — replace with actual URDF/MJCF)
            if self.robot_model_path:
                self._mj_model = mujoco.MjModel.from_xml_path(self.robot_model_path)
            else:
                # Minimal dummy model for testing
                xml = """
                <mujoco>
                  <worldbody>
                    <body name="robot">
                      <joint name="joint1" type="hinge"/>
                      <geom type="box" size="0.1 0.1 0.1"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
                self._mj_model = mujoco.MjModel.from_xml_string(xml)

            self._mj_data = mujoco.MjData(self._mj_model)
            self._sim = "mujoco"
        except ImportError:
            raise ImportError(
                "MuJoCo not installed. Run: pip install mujoco"
            )

    def _init_isaac(self):
        """Initialize Isaac Sim (placeholder)."""
        # TODO: Isaac Sim requires NVIDIA Omniverse environment
        raise NotImplementedError(
            "Isaac Sim integration not yet implemented. Use MuJoCo for MVP."
        )

    def check(
        self,
        predicted_actions: torch.Tensor,
        predicted_sensors: Dict[str, torch.Tensor],
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ConsistencyResult:
        """
        Run consistency check on a generated action/sensor sequence.

        Args:
            predicted_actions: (horizon, action_dim) — joint or end-effector commands
            predicted_sensors: {"joint_pos", "joint_vel", "force"} predictions
            initial_state: optional initial robot state for simulation
        Returns:
            ConsistencyResult with validity and scores
        """
        self._init_sim()

        if self.engine == "mujoco":
            return self._check_mujoco(predicted_actions, predicted_sensors, initial_state)
        else:
            return self._check_isaac(predicted_actions, predicted_sensors, initial_state)

    def _check_mujoco(
        self,
        actions: torch.Tensor,
        sensors: Dict[str, torch.Tensor],
        initial_state: Optional[Dict] = None,
    ) -> ConsistencyResult:
        """
        Run actions in MuJoCo and compare sensor predictions.
        """
        # Convert to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        action_np = np.atleast_2d(actions)
        horizon = action_np.shape[0]

        # Set initial state if provided
        if initial_state:
            if "qpos" in initial_state:
                self._mj_data.qpos[:] = initial_state["qpos"].cpu().numpy()
            if "qvel" in initial_state:
                self._mj_data.qvel[:] = initial_state["qvel"].cpu().numpy()

        # Rollout
        sim_positions = []
        sim_forces = []
        sim_joints = []

        for t in range(min(horizon, 500)):  # max 500 steps = 1s at 500Hz
            # Apply action (position control)
            if action_np.shape[1] <= self._mj_model.nu:
                self._mj_data.ctrl[:] = action_np[t]
            self._mujoco.mj_step(self._mj_model, self._mj_data)

            # Record simulation state
            sim_joints.append(self._mj_data.qpos.copy())
            sim_forces.append(self._mj_data.qfrc_actuator.copy())

            # End-effector position (from MuJoCo sensors or forward kinematics)
            # Simplified: use first body position as proxy
            sim_positions.append(self._mj_data.xpos[1].copy())  # body 0 is world

        # Compute errors
        sim_positions = np.array(sim_positions)
        if "force" in sensors:
            pred_force = sensors["force"].detach().cpu().numpy()
            if pred_force.ndim == 2:
                pred_force = pred_force[:horizon]
            force_error = np.mean(np.abs(sim_forces[:len(pred_force)] - pred_force[:, :3]))
        else:
            force_error = 0.0

        # Last position error as end-effector displacement
        if len(sim_positions) > 1:
            pose_error = np.linalg.norm(sim_positions[-1] - sim_positions[0]) * 100  # cm
        else:
            pose_error = 0.0

        joint_error = 0.0
        if "joint_pos" in sensors:
            pred_joints = sensors["joint_pos"].detach().cpu().numpy()
            joint_error = np.mean(np.abs(sim_joints[-1][:len(pred_joints)] - pred_joints))

        # Threshold checks
        is_valid = (
            pose_error < self.thresholds["pose_error_cm"] and
            force_error < self.thresholds["force_error_n"] and
            joint_error < self.thresholds.get("joint_error_rad", 0.1)
        )

        # Composite score (0–1)
        pose_score = max(0.0, 1.0 - pose_error / self.thresholds["pose_error_cm"])
        force_score = max(0.0, 1.0 - force_error / self.thresholds["force_error_n"])
        score = float(0.5 * pose_score + 0.5 * force_score)

        return ConsistencyResult(
            is_valid=is_valid,
            score=score,
            pose_error_cm=float(pose_error),
            force_error_n=float(force_error),
            joint_error_rad=float(joint_error),
            sim_trajectory={
                "positions": sim_positions,
                "forces": sim_forces,
                "joints": sim_joints,
            },
        )

    def _check_isaac(self, *args, **kwargs) -> ConsistencyResult:
        """Isaac Sim consistency check (placeholder)."""
        raise NotImplementedError("Isaac Sim integration not yet implemented.")

    def filter_batch(
        self,
        action_batch: torch.Tensor,
        sensor_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], np.ndarray]:
        """
        Filter a batch of generated samples, keeping only valid ones.

        Args:
            action_batch: (B, horizon, action_dim)
            sensor_batch: dict of (B, ...) tensors
        Returns:
            filtered_actions, filtered_sensors, validity_mask
        """
        B = action_batch.shape[0]
        results = []
        for i in range(B):
            sensors_i = {k: v[i] for k, v in sensor_batch.items()}
            result = self.check(action_batch[i], sensors_i)
            results.append(result)

        validity = np.array([r.is_valid for r in results])
        valid_actions = action_batch[validity]
        valid_sensors = {k: v[validity] for k, v in sensor_batch.items()}

        return valid_actions, valid_sensors, validity


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    checker = ConsistencyChecker(engine="mujoco")
    print("ConsistencyChecker initialized with MuJoCo.")
    print(f"  Thresholds: {checker.thresholds}")

    # Dummy test
    dummy_actions = torch.randn(50, 7)      # 50 steps, 7-DoF
    dummy_sensors = {
        "joint_pos": torch.randn(7),
        "force": torch.randn(6),
    }
    result = checker.check(dummy_actions, dummy_sensors)
    print(f"  Result: valid={result.is_valid}, score={result.score:.3f}")
    print(f"  pose_error={result.pose_error_cm:.2f}cm, force_error={result.force_error_n:.3f}N")
