# Configs — 配置说明

## 配置文件

| 文件 | 说明 |
|------|------|
| [default.yaml](default.yaml) | 默认训练配置，所有参数集中管理 |

## 配置结构

```yaml
model:           # 模型架构参数
  visual_encoder: "resnet50"
  world_latent_dim: 256
  action_dim: 7
  action_horizon: 16
  diffusion_steps: 100
  ...

training:        # 训练超参
  batch_size: 64
  learning_rate: 1.0e-4
  max_steps: 1_000_000
  loss_weights:
    action_mse: 1.0
    joint_mse: 0.5
    force_mae: 0.3
  ...

data:            # 数据配置
  real_dataset: "bridge_v2"
  real_data_path: "./data/real/"
  image_size: [224, 224]
  ...

simulation:      # 仿真配置
  engine: "mujoco"
  consistency_thresholds:
    pose_error_cm: 5.0
    force_error_n: 0.5
  ...

hardware:        # 硬件配置
  device: "cuda"
  gpu_ids: [0]
  mixed_precision: "fp16"
  ...
```

## 自定义配置

```bash
# 复制默认配置
cp configs/default.yaml configs/my_experiment.yaml

# 修改参数后运行
python scripts/train.py --config configs/my_experiment.yaml
```
