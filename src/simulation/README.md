# Simulation — 仿真验证模块

## 模块概览

| 文件 | 类 | 功能 |
|------|-----|------|
| [consistency.py](consistency.py) | `ConsistencyChecker` | 物理仿真一致性检验 |
| [consistency.py](consistency.py) | `ConsistencyResult` | 检验结果数据结构 |

## 一致性检验流程

```
生成的动作+传感器 ──► MuJoCo 仿真执行 ──► 仿真轨迹+状态
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │  误差计算       │
                                      │               │
                                      │  位姿误差 (cm)  │
                                      │  力误差 (N)     │
                                      │  关节误差 (rad) │
                                      └───────┬───────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │  阈值过滤       │
                                      │               │
                                      │  位姿 < 5cm    │
                                      │  力 < 0.5N     │
                                      │  关节 < 0.1rad │
                                      └───────┬───────┘
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                               ✅ 合格数据           ❌ 丢弃
                              (加入合成数据集)
```

## 仿真引擎

| 引擎 | 状态 | 说明 |
|------|------|------|
| **MuJoCo** | ✅ 已实现 | 开源免费，Python 原生支持，适合 MVP |
| **Isaac Sim** | ⏳ 待实现 | NVIDIA 企业级，更精确但需 Omniverse |

## 使用方法

```python
from src.simulation import ConsistencyChecker

checker = ConsistencyChecker(
    engine="mujoco",
    robot_model_path="data/robot/franka.xml",  # 可选
    thresholds={"pose_error_cm": 5.0, "force_error_n": 0.5},
)

# 单条检验
result = checker.check(predicted_actions, predicted_sensors)
print(f"valid={result.is_valid}, score={result.score:.3f}")

# 批量过滤
valid_actions, valid_sensors, mask = checker.filter_batch(actions, sensors)
```

## 误差指标

| 指标 | 阈值 | 计算方式 |
|------|------|----------|
| `pose_error_cm` | < 5.0 | 仿真末端位置与预期轨迹的欧氏距离 |
| `force_error_n` | < 0.5 | 仿真接触力与预测力的 MAE |
| `joint_error_rad` | < 0.1 | 仿真关节角与预测角的 RMSE |
| `score` | > 0.8 | 0.5×位姿分数 + 0.5×力分数（0-1） |

## 待完成

- [ ] 加载真实机器人 URDF/MJCF 模型
- [ ] 载入任务场景（桌面、物体）
- [ ] Isaac Sim 集成
- [ ] 域随机化（噪声注入、物理参数扰动）

## 参考

- MuJoCo: [MuJoCo](https://mujoco.org/) (Todorov et al., 2012)
- Isaac Sim: [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
