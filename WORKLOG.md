# RDAE 工作日志

> 每次推送代码或有阶段性成果时更新。记录做了什么、改了什么、待办事项、下一步方向。

---

## 2026-06-04 — 架构 v2 实现完成（对标 SOTA）

### 🎯 工作内容
1. **世界编码器 v2 重写**：冻结 ViT-B + Perceiver Resampler (64 queries) + 8层因果 Transformer + [CLS][FRS][INV] 三专用 token
2. **动作解码器 v2 升级**：DDIM 10步采样（替代 DDPM 100步），可变卷积核 + 扩张卷积
3. **传感器解码器 v2 重写**：Temporal Transformer + ContactDetector（接触二分类）+ 物理一致性损失
4. **训练脚本 v2**：支持 pretrain / finetune / validate 三阶段，混合精度训练，梯度累积

### 📦 交付成果
| 文件 | 行数 | 变更 |
|------|:---:|------|
| `src/models/encoder.py` | 375 → **497** | 完全重写 |
| `src/models/sensor_decoder.py` | 180 → **374** | 完全重写 |
| `src/models/action_decoder.py` | 210 → **328** | DDIM + 优化 |
| `scripts/train.py` | 193 → **277** | 三阶段训练 |
| `configs/default.yaml` | 87 → **107** | v2 配置 |

### 🔧 关键改进
| 维度 | v1 | v2 |
|------|----|----|
| 视觉编码 | ResNet-50 → 单向量 | ViT-B 冻结 → 197 patch tokens |
| Token 压缩 | 无 | Perceiver Resampler (64 tokens) |
| 融合 | 2层, 3 tokens | 8层因果 Transformer, 数百 tokens |
| 逆动力学 | CLS token 共享 | [INV] 专用 token |
| 辅助任务 | 无 | [FRS] 未来帧预测 |
| 动作采样 | DDPM 100步 (~500ms) | DDIM 10步 (~50ms) |
| 传感器 | mean_pool + MLP | Temporal Transformer + ContactDetector |
| 物理约束 | 无 | Lagrangian 一致性损失 |

### 📋 待办事项
- [ ] 安装环境 + 下载预训练 ViT 权重
- [ ] 下载 Ego4D 子集做视频预测预训练
- [ ] 下载 BridgeData V2 做多任务微调
- [ ] 端到端训练跑通并对比 v1 性能

### 🧭 下一步工作方向
1. 环境搭建 + 数据准备
2. 视频预训练验证
3. 端到端微调 + 仿真一致性评估


## 2026-06-03（更新3）— 技术调研完成 & 架构方案确定

### 🎯 工作内容
1. 调研 **18 个权威开源项目/论文**，覆盖 GitHub + HuggingFace + PapersWithCode + arXiv
2. 产出 447 行调研报告 `RESEARCH.md`，包含架构对比、差距分析、修正方案
3. 确定修正后的 RDAE 架构方案（对标 GR-1 + Seer + FLARE）
4. 产出一份 `DESIGN_REVIEW.md` 设计评审报告

### 📦 交付成果
| 文档 | 内容 |
|------|------|
| `RESEARCH.md` | 18 个项目调研，三大模块 SOTA 对比，HuggingFace/PapersWithCode 发现 |
| `DESIGN_REVIEW.md` | 当前设计 5 个问题 + 优化方案 + 优先级排序 |
| `STAGE1.md` + 6 个模块 README | 完整文档体系 |

### 🔍 关键发现
1. **世界编码器**：所有人都在用"小视觉编码（冻结 ViT）+ 大 Transformer"，不是 ResNet + 小 Transformer
2. **动作解码器**：当前 1D CNN UNet + FiLM 方向正确，只需 DDPM→DDIM + 小优化
3. **传感器解码器**：无直接开源参照（RDAE 的设定相对新颖），需参照 Seer 的 [INV] token + 物理约束
4. **预训练策略**：视频预测是最有效的预训练任务（GR-1 +76%），MCR 证实机器人数据对比预训练 >> ImageNet

### 🧭 下一步工作方向
1. **立即执行**：按照调研结论修改编码器和传感器解码器架构
2. **预训练**：下载 Ego4D 子集做视频预测预训练
3. **微调**：在 BridgeData V2 上端到端微调


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
