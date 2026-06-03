# Utils — 工具模块

## 模块概览

| 文件 | 功能 | 说明 |
|------|------|------|
| [config.py](config.py) | 配置加载/保存 | YAML 读写 |
| [metrics.py](metrics.py) | 评估指标 | 位姿误差、关节RMSE、力MAE、一致性分数 |

## 使用方法

```python
from src.utils import load_config
from src.utils.metrics import compute_pose_error, compute_joint_rmse, compute_force_error

# 加载配置
cfg = load_config("configs/default.yaml")

# 计算误差
pos_err = compute_pose_error(pred_pose, gt_pose)       # 欧氏距离 (m)
joint_err = compute_joint_rmse(pred_joints, gt_joints)  # RMSE (rad)
force_err = compute_force_error(pred_force, gt_force)   # MAE (N)
```
