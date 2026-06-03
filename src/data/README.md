# Data — 数据管道模块

## 模块概览

| 文件 | 功能 | 说明 |
|------|------|------|
| [dataset.py](dataset.py) | 统一数据加载器 | 支持 BridgeData V2 / DROID / 自定义格式 |
| [preprocessing.py](preprocessing.py) | 数据预处理 | RGB 缩放/归一化、关节状态归一化 |

## 支持的数据集

### BridgeData V2

- **规模**: 60k 轨迹，24 环境，13 技能
- **格式**: HDF5 (`.hdf5`)
- **键值**:
  - `observations/images0`: RGB 图像 (H, W, C)
  - `observations/state`: 关节+夹爪状态
  - `actions`: 动作序列
  - `language` (可选): 语言指令

```python
dataset = RobotDataset(
    data_path="data/real/bridge_v2/",
    dataset_type="bridge_v2",
    image_size=(224, 224),
    frame_stack=2,
    action_horizon=16,
)
```

### DROID

- **规模**: 350 小时，76k 轨迹，564 场景
- **格式**: MCAP (`.mcap`)
- **状态**: ⚠️ MCAP 解析待实现

### 自定义格式

- **格式**: NPZ (`.npz`) 或 manifest.json

```python
# NPZ 文件需包含键: images, state, actions
dataset = RobotDataset(
    data_path="data/real/custom/",
    dataset_type="custom",
)
```

## 数据格式规范

### 样本结构

```python
sample = {
    "images":   torch.Tensor,   # (T, C, H, W) 或 (C, H, W)
    "state":    torch.Tensor,   # (state_dim,) 关节角度+速度
    "actions":  torch.Tensor,   # (horizon, action_dim) 未来动作序列
    "language": str,            # 语言指令（可选）
    # 以下为可选传感器数据
    "joint_pos": torch.Tensor,  # (joint_dim,)
    "joint_vel": torch.Tensor,  # (joint_dim,)
    "force":    torch.Tensor,   # (6,) 力+力矩
}
```

## 预处理流程

```
原始数据                   预处理后
─────────                ─────────
RGB [0,255], 480×640  →  [0,1], 224×224, ImageNet归一化
关节角度 (rad)         →  零均值单位方差
动作 (末端位姿差)       →  零均值单位方差
```

## 数据存放约定

```
data/
├── real/                    ← 真实机器人数据（.gitignore 忽略）
│   ├── bridge_v2/
│   │   ├── *.hdf5
│   │   └── stats_train.npz
│   ├── droid/
│   │   └── *.mcap
│   └── custom/
│       ├── manifest.json
│       └── *.npz
└── synthetic/               ← RDAE 生成的合成数据（Stage 2+）
    └── ...
```

## 快速验证

```bash
python src/data/dataset.py
```

## 参考

- BridgeData V2: [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) (Walke et al., CoRL 2024)
- DROID: [DROID](https://droid-dataset.github.io/) (Khazatsky et al., 2024)
- LeRobot: [Video Encoding for Robotics](https://huggingface.co/blog/lerobot-video-encoding) (Alibert et al., 2024)
