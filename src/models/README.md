# Models — 模型模块 v2

> 2026-06-03 架构升级 | 对标 GR-1 + Seer + Diffusion Policy

## 模块概览

| 文件 | 类 | 功能 | 架构 |
|------|-----|------|------|
| [encoder.py](encoder.py) | `MultiModalWorldEncoder` | 融合视觉+状态 | ViT + Perceiver + 8层因果Transformer |
| [encoder.py](encoder.py) | `FrozenViTEncoder` | 冻结 ViT 视觉编码 | ViT-B/16 patch tokens |
| [encoder.py](encoder.py) | `PerceiverResampler` | 压缩 patch tokens | 64 learnable queries + Cross-Attention |
| [encoder.py](encoder.py) | `CausalWorldTransformer` | 时序融合+逆动力学 | 8层 GPT-2 风格因果 Transformer |
| [action_decoder.py](action_decoder.py) | `ActionDecoder` | 动作序列生成 | 1D CNN UNet + FiLM + DDIM (10步) |
| [sensor_decoder.py](sensor_decoder.py) | `SensorDecoder` | 传感器重建 | Temporal Transformer + ContactDetector |
| [sensor_decoder.py](sensor_decoder.py) | `ContactDetector` | 接触事件检测 | 双向 GRU + 二分类 |

## v2 架构图

```
RGB (T, 224, 224) ──► Frozen ViT-B ──► (T, 197 patches, 768)
                                              │
                                              ▼
                                     Perceiver Resampler
                                     (64 query tokens)
                                              │
                                              ▼
                                     (T, 64 tokens, 384)
                                              │
              ┌───────────────────────────────┘
              ▼
State (T,14) ──► MLP ──► (T, 1 token, 384)
              │
              ▼
  ┌───────────────────────────────┐
  │  Causal World Transformer     │  [IMG tokens | STATE tokens | CLS | FRS | INV]
  │  8 layers, 384-d, 12 heads    │
  └───┬───────┬───────┬───────────┘
      ▼       ▼       ▼
  z_w(256) FRS_pred INV_feat
      │               │
      ▼               ▼
┌──────────────┐ ┌──────────────────┐
│ ActionDecoder│ │  SensorDecoder   │
│ DDIM 10步   │ │  Temporal Trans.  │
│ → 动作序列   │ │  → 关节/力/接触   │
└──────────────┘ └──────────────────┘
```

## v1 → v2 改动

| 模块 | v1 | v2 | 来源 |
|------|----|----|------|
| 视觉编码 | ResNet-50 (单向量) | ViT-B 冻结 (patch tokens) | GR-1, MCR |
| Token压缩 | 无 | Perceiver Resampler (64 tokens) | GR-1, Seer |
| 融合Transformer | 2层, 3 tokens | 8层因果, 数百 tokens | GR-1, Octo |
| 时序建模 | 外部 frame_stack | 因果注意力 + 时序PE | GR-1, Seer |
| 逆动力学 | CLS token | [INV] 专用 token | Seer |
| 辅助任务 | 无 | 未来帧预测 [FRS] token | Seer, GR-1 |
| 动作采样 | DDPM 100步 | DDIM 10步 (~10× faster) | Diffusion Policy |
| 传感器 | mean_pool + MLP | Temporal Transformer | Seer |
| 接触建模 | 无 | ContactDetector (二分类) | FILIC |
| 物理约束 | 无 | Lagrangian 一致性损失 | LIP4RobotID |

## 参数量

| 模块 | 参数 | 可训练 |
|------|------|:---:|
| ViT-B/16 | 86M | ❌ |
| Perceiver | ~2M | ✅ |
| Transformer (8层) | ~28M | ✅ |
| Action Decoder | ~1.5M | ✅ |
| Sensor Decoder | ~4M | ✅ |
| **总计** | **~121M** | **~36M** |

## 快速验证

```bash
python src/models/encoder.py         # 编码器测试
python src/models/action_decoder.py  # DDIM vs DDPM 速度对比
python src/models/sensor_decoder.py  # 传感器+接触检测测试
```

## 参考

- **GR-1**: ByteDance, ICLR 2024. MAE + Perceiver + Causal Transformer
- **Seer**: Shanghai AI Lab, ICLR 2025 Oral. [FRS] + [INV] tokens
- **Octo**: Berkeley/Stanford/CMU, RSS 2024. Modular Transformer + Diffusion
- **MCR**: DROID contrastive pretraining >> ImageNet (2024)
- **Diffusion Policy**: Chi et al., RSS 2023. 1D CNN UNet + DDIM
- **FILIC**: Joint torque → force estimation (2025)
- **LIP4RobotID**: MERL, Lagrangian physics constraints
