# Stage 1 — MVP 基线模型

> 周期：0-3 个月 | 分支：`stage-1` | 依赖：白皮书 §147-157

## 目标

搭建 RDAE 最小可行产品（MVP）的数据管道与基线模型：
- 加载真实机器人数据集（BridgeData V2 / DROID）
- 训练多模态世界编码器（视觉 + 关节状态 → 世界潜空间）
- 训练动作解码器（扩散策略，从潜空间生成控制序列）
- 训练传感器解码器（从潜空间+动作重建关节/力/触觉）
- 在 MuJoCo 仿真中验证生成的合成数据质量

## 目录结构

```
stage-1/
├── STAGE1.md                     ← 本文档
├── WORKLOG.md                    ← 工作日志
├── requirements.txt              ← 环境依赖
├── pyproject.toml                ← 项目配置
├── configs/
│   ├── default.yaml              ← 训练/模型/数据/仿真全配置
│   └── README.md                 ← 配置说明
├── src/
│   ├── models/
│   │   ├── encoder.py            ← 多模态世界编码器
│   │   ├── action_decoder.py     ← 扩散动作解码器
│   │   ├── sensor_decoder.py     ← 传感器重建模块
│   │   └── README.md             ← 模型说明
│   ├── data/
│   │   ├── dataset.py            ← 统一数据加载器
│   │   ├── preprocessing.py      ← 数据预处理
│   │   └── README.md             ← 数据管道说明
│   ├── simulation/
│   │   ├── consistency.py        ← MuJoCo 一致性检验
│   │   └── README.md             ← 仿真模块说明
│   └── utils/
│       ├── config.py             ← YAML 配置加载
│       ├── metrics.py            ← 误差指标计算
│       └── README.md             ← 工具说明
├── scripts/
│   ├── train.py                  ← 训练入口
│   ├── evaluate.py               ← 评估入口
│   └── README.md                 ← 脚本说明
└── tests/                        ← 单元测试
```

## 快速开始

```bash
# 1. 环境
pip install -r requirements.txt
pip install mujoco

# 2. 下载数据（选一个）
# BridgeData V2: https://rail-berkeley.github.io/bridgedata/
# DROID: https://droid-dataset.github.io/

# 3. 放置数据
mkdir -p data/real
# 将 .hdf5 或 .mcap 文件放入 data/real/

# 4. 验证各模块
python src/models/encoder.py        # 编码器测试
python src/models/action_decoder.py # 解码器测试
python src/models/sensor_decoder.py # 传感器测试
python src/simulation/consistency.py # 仿真测试

# 5. 完整训练
python scripts/train.py --config configs/default.yaml --debug

# 6. 评估
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_step0010000.pt
```

## 模型架构速览

```
                    互联网视频帧
                    (Stage 2 接入)
                         │
                         ▼
┌──────────────────────────────────────────────┐
│           多模态世界编码器                      │
│                                              │
│  RGB ──► ResNet/ViT ──┐                     │
│                        ├──► Transformer ──► z_w (256维)
│  关节 ──► MLP ────────┘                     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌──────────────┐              ┌────────────────┐
│  动作解码器    │              │  传感器解码器    │
│  Diffusion    │              │  3层 MLP        │
│  Policy π₀    │              │  多任务回归      │
│              │              │                │
│  输出: 动作序列│              │  输出: 关节/力   │
└──────┬───────┘              └──────┬─────────┘
       │                             │
       └──────────┬──────────────────┘
                  ▼
       ┌──────────────────┐
       │  MuJoCo 仿真验证   │
       │  一致性过滤        │
       │  (位姿<5cm, 力<0.5N)│
       └──────────────────┘
```

## 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 动作 MSE | < 0.01 | 预测动作 vs 真实动作 |
| 关节 RMSE | < 0.05 rad | 预测关节角度 vs 真实 |
| 力 MAE | < 0.5 N | 预测力 vs 真实 |
| 一致性分数 | > 0.8 | 仿真验证综合评分 |

## 下一步

- [ ] 下载数据并验证数据管道
- [ ] 端到端训练首次跑通
- [ ] 生成第一批合成数据样本
- [ ] 进入 Stage 2：互联网视频管道
