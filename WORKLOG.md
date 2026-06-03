# RDAE 工作日志

> 每次推送代码或有阶段性成果时更新。记录做了什么、改了什么、待办事项、下一步方向。

---

## 2026-06-03（更新2）— 模块文档体系建立

### 🎯 工作内容
1. 创建 `STAGE1.md`——Stage-1 总体说明（目标、目录、快速开始、架构速览）
2. 为 6 个子模块各写一份 `README.md`，覆盖架构、用法、API、参考

### 📦 交付成果
| 文档 | 路径 | 内容 |
|------|------|------|
| Stage-1 总览 | `STAGE1.md` | 目标、目录、快速开始、架构图、指标 |
| 模型说明 | `src/models/README.md` | 编码器/解码器架构、参数量、代码示例 |
| 数据管道说明 | `src/data/README.md` | 数据集支持、格式规范、预处理流程 |
| 仿真模块说明 | `src/simulation/README.md` | 一致性流程、引擎对比、API |
| 工具模块说明 | `src/utils/README.md` | 配置加载、误差指标 |
| 脚本说明 | `scripts/README.md` | train.py / evaluate.py 参数和流程 |
| 配置说明 | `configs/README.md` | YAML 结构、自定义方法 |

### 🔧 改动记录
- 新增 7 个文档文件（STAGE1.md + 6 个模块 README.md）
- 每个 README 包含架构图、代码示例、参数说明、参考文献

### 📋 待办事项
- [x] 推送代码到 GitHub ✅
- [x] 各模块独立说明文档 ✅
- [ ] 安装 Python 环境：`pip install -r requirements.txt`
- [ ] 下载 BridgeData V2 或 DROID 数据集
- [ ] 端到端训练测试：`python scripts/train.py --debug`

### 🧭 下一步工作方向
1. 环境安装 + 数据下载
2. 首次训练跑通
3. 进入 Stage 2：互联网视频管道


## 2026-06-03（更新）— 推送成功 & SSH 配置

### 🎯 工作内容
1. HTTPS 连接 GitHub 超时，切换为 SSH 协议后推送成功
2. `stage-1` 分支已上线 GitHub（21 个文件）

### 🔧 改动记录
- 远程地址从 HTTPS 切换为 SSH：`git@github.com:2453541752/RDAE-Robot-Data-Amplification-Engine.git`
- HTTPS 在国内直连不稳定，SSH 可正常使用

### 📋 待办事项
- [x] 推送代码到 GitHub ✅

---

## 2026-06-03 — 项目启动 & MVP Stage-1 骨架搭建

### 🎯 工作内容
1. **项目初始化**：创建 GitHub 仓库，上传 RDAE v1.0 白皮书作为 README
2. **分支策略建立**：`main`（稳定）← `stage-1`（MVP 阶段1开发分支）
3. **项目骨架**：完整 Python 项目结构，20 个文件，~2100 行代码

### 📦 交付成果
| 模块 | 文件 | 功能 |
|------|------|------|
| 配置 | `requirements.txt`, `pyproject.toml`, `configs/default.yaml` | 依赖管理、项目元信息、训练超参 |
| 世界编码器 | `src/models/encoder.py` | ResNet-50/ViT-B + 2层 Transformer → 256维 z_w |
| 动作解码器 | `src/models/action_decoder.py` | Diffusion Policy (100步DDPM)，16步动作预测 |
| 传感器解码器 | `src/models/sensor_decoder.py` | 3层 MLP 多任务回归（关节/力/触觉） |
| 数据管道 | `src/data/dataset.py`, `preprocessing.py` | BridgeData V2 / DROID / 自定义格式加载 |
| 仿真验证 | `src/simulation/consistency.py` | MuJoCo 一致性检验（位姿<5cm, 力<0.5N） |
| 训练脚本 | `scripts/train.py` | 完整训练循环 + TensorBoard 日志 |
| 评估脚本 | `scripts/evaluate.py` | MSE/RMSE/一致性分数计算 |
| 工具函数 | `src/utils/config.py`, `metrics.py` | YAML 配置加载、误差指标 |
| 工作日志 | `WORKLOG.md` | 本文档 |

### 🔧 改动记录
- 新建 GitHub 仓库 `2453541752/RDAE-Robot-Data-Amplification-Engine`
- `deep-research-report.md` → `README.md`（仓库首页展示）
- 上传两张项目架构图 PNG
- 创建 `stage-1` 分支
- 搭建完整项目目录结构
- 编写所有 MVP Stage-1 模块骨架代码
- 创建 `.gitignore`（修复 `data/` 误匹配 `src/data/` 问题）

### 📋 待办事项
- [ ] **网络恢复后**：`git push -u origin stage-1`
- [ ] 安装 Python 环境：`pip install -r requirements.txt`
- [ ] 安装 MuJoCo：`pip install mujoco`
- [ ] 下载 BridgeData V2 或 DROID 数据集到 `data/real/`
- [ ] 验证各模块：运行 `if __name__ == "__main__"` 测试代码
- [ ] 端到端训练测试：`python scripts/train.py --debug`
- [ ] 准备机器人 URDF/MJCF 模型文件用于仿真
- [ ] 编写单元测试 `tests/`

### 🧭 下一步工作方向
1. **短期（本周）**：环境配置 + 数据下载 + 模块单元测试
2. **中期（Stage-1, 0-3月）**：端到端训练跑通 → 仿真验证 → 小规模合成数据生成
3. **长期（Stage-2+, 3-12月）**：互联网视频管道 → 百万级数据扩增 → VLA 策略评估

---

## 模板（后续每次更新复制此格式）

```
## YYYY-MM-DD — 简短标题

### 🎯 工作内容
1. xxx
2. xxx

### 📦 交付成果
- xxx

### 🔧 改动记录
- xxx

### 📋 待办事项
- [ ] xxx
- [x] xxx（已完成）

### 🧭 下一步工作方向
1. xxx
2. xxx
```
