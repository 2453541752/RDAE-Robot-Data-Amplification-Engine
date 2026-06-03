# Scripts — 脚本入口

## 脚本列表

| 文件 | 功能 | 用法 |
|------|------|------|
| [train.py](train.py) | 完整训练流程 | `python scripts/train.py --config configs/default.yaml` |
| [evaluate.py](evaluate.py) | 模型评估 | `python scripts/evaluate.py --checkpoint checkpoints/xxx.pt` |

## train.py — 训练脚本

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/default.yaml` |
| `--resume` | 从 checkpoint 恢复 | `None` |
| `--debug` | 调试模式（少量数据快速验证） | `False` |

### 训练流程

```
加载配置 → 初始化模型 → 加载数据 → 循环训练
                                    │
                    ┌───────────────┘
                    ▼
            对每个 batch:
              encoder(images, state) → z_w
              action_decoder(z_w, gt_actions) → noise_pred
              sensor_decoder(z_w, pred_actions) → sensors
              计算 loss = λ₁·action_mse + λ₂·sensor_loss
              反向传播 → 更新参数
                    │
                    ├─ 每 100 步 → 日志输出
                    ├─ 每 5000 步 → 验证
                    └─ 每 10000 步 → 保存 checkpoint
```

### 输出

- `checkpoints/checkpoint_step*.pt` — 模型权重
- `logs/` — TensorBoard 日志

## evaluate.py — 评估脚本

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/default.yaml` |
| `--checkpoint` | 模型路径（必填） | — |
| `--output` | 结果 JSON 路径 | `eval_results.json` |

### 输出指标

```json
{
  "action_mse":      {"mean": 0.008, "std": 0.002},
  "joint_rmse":      {"mean": 0.032, "std": 0.010},
  "force_mae":       {"mean": 0.215, "std": 0.080},
  "consistency_score": {"mean": 0.87, "std": 0.05}
}
```
