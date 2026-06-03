# Models — 模型模块

## 模块概览

| 文件 | 类 | 功能 | 架构 |
|------|-----|------|------|
| [encoder.py](encoder.py) | `MultiModalWorldEncoder` | 融合视觉+状态，输出世界潜空间 | ResNet/ViT + 2层 Transformer |
| [encoder.py](encoder.py) | `VisualEncoder` | 视觉特征提取 | ResNet-50 或 ViT-B/16 |
| [encoder.py](encoder.py) | `StateEncoder` | 机器人本体状态编码 | 3层 MLP |
| [action_decoder.py](action_decoder.py) | `ActionDecoder` | 从潜空间生成控制序列 | Diffusion Policy (DDPM) |
| [action_decoder.py](action_decoder.py) | `ConditionalUNet1D` | 条件去噪网络 | 1D UNet + FiLM 条件 |
| [sensor_decoder.py](sensor_decoder.py) | `SensorDecoder` | 重建关节/力/触觉读数 | 3层 MLP + 多头回归 |

## 架构详情

### 1. MultiModalWorldEncoder

```
输入: images (B,3,224,224) + state (B,14)
      │                          │
      ▼                          ▼
 VisualEncoder              StateEncoder
 (ResNet-50 → 512)          (MLP → 512)
      │                          │
      ▼                          ▼
 [CLS][VIS][STATE]   ← Token 序列拼接
      │
      ▼
 2层 TransformerEncoder (d=512, heads=8)
      │
      ▼
 输出: z_w (B, 256)   ← 世界潜空间
```

### 2. ActionDecoder (Diffusion Policy)

```
训练:  z_w + 真实动作 ──► 加噪 ──► UNet 预测噪声 ──► MSE Loss
推理:  z_w + 随机噪声 ──┬── 第100步去噪 ──► 预估x0
                        ├── 第99步去噪 ──► ...
                        └── 第1步去噪 ──► 干净动作序列

输出: actions (B, 7, 16)  ← 7维末端动作 × 16步预测
```

扩散步数: 100，噪声调度: Cosine，去噪器: 1D CONV UNet + FiLM

### 3. SensorDecoder

```
输入: z_w (256) ⊕ action (7)
      │
      ▼
 共享主干: 3层 MLP (512→256→128)
      │
      ├──► joint_pos_head  → (7,) 关节角度
      ├──► joint_vel_head  → (7,) 关节速度
      ├──► force_head      → (6,) 力+力矩
      └──► tactile_head    → (N,) 触觉（可选）

损失: λ₁·MSE(joint) + λ₂·L1(force)
```

## 参数量估算

| 模块 | 参数 |
|------|------|
| VisualEncoder (ResNet-50) | ~23.5M |
| StateEncoder (3层 MLP) | ~0.3M |
| Transformer (2层) | ~4.2M |
| ActionDecoder (UNet) | ~1.5M |
| SensorDecoder (MLP) | ~0.5M |
| **总计** | **~30M** |

## 使用方法

```python
from src.models import MultiModalWorldEncoder, ActionDecoder, SensorDecoder

# 初始化
encoder = MultiModalWorldEncoder(visual_backbone="resnet50", world_latent_dim=256)
action_dec = ActionDecoder(action_dim=7, action_horizon=16, cond_dim=256)
sensor_dec = SensorDecoder(world_latent_dim=256, action_dim=7)

# 前向传播
z_w = encoder(images, joint_state)                    # (B, 256)
pred_actions = action_dec(z_w, gt_actions)             # 训练: 返回噪声
pred_actions = action_dec(z_w, None)                   # 推理: 生成动作
sensors = sensor_dec(z_w, pred_actions)                # 重建传感器
```

## 参考

- ResNet: [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He et al., 2016)
- ViT: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)
- Diffusion Policy: [Diffusion Policy](https://arxiv.org/abs/2303.04137) (Chi et al., 2023)
- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- AMPLIFY: [Actionless Motion Priors](https://arxiv.org/abs/2404.12345) (Collins et al., 2024)
