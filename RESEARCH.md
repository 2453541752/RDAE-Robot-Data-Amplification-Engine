# RDAE Stage-1 技术调研报告

> 2026-06-03 | 调研 15+ 个权威开源项目/论文，对标三大核心模块

---

## 目录

- [1. 多模态世界编码器调研](#1-多模态世界编码器调研)
- [2. 动作解码器调研](#2-动作解码器调研)
- [3. 逆动力学/传感器重建调研](#3-逆动力学传感器重建调研)
- [4. 当前设计差距分析](#4-当前设计差距分析)
- [5. 修正后的架构建议](#5-修正后的架构建议)

---

## 1. 多模态世界编码器调研

### 1.1 主流方案对比

| 项目 | 机构 | 视觉编码器 | 融合方式 | 时序建模 | 预训练 | 代码 |
|------|------|-----------|---------|---------|--------|:---:|
| **GR-1** | ByteDance | MAE ViT-B + Perceiver Resampler | GPT-2 风格因果 Transformer | ✅ 因果注意力 | Ego4D 视频预测 | ✅ |
| **Octo** | Berkeley/Stanford/CMU | 浅层 CNN (SmallStem16) | Block-wise Masked Transformer | ✅ 逐块掩码 | Open X-Embodiment | ✅ |
| **Seer** | 上海 AI Lab | ViT + Perceiver Resampler | GPT 风格 Transformer + [FRS][INV] tokens | ✅ 因果注意力 | 端到端训练 | ✅ |
| **π₀** | Physical Intelligence | SigLIP ViT (PaliGemma) | MoE 双流 Transformer | ✅ 块因果掩码 | VLM + 机器人数据 | ✅ |
| **R3M** | Meta/Facebook | ResNet-50 | —（仅视觉表征） | ❌ | 视频时间对比学习 | ✅ |
| **AMPLIFY** | Georgia Tech | 关键点检测 + FSQ 量化 | 自回归 Transformer | ✅ Motion tokens | 动作自由视频 | ❌ |
| **ViPRA** | — | 视频预测模型 | 光流一致性校正 | ✅ | 视频预测 | ❌ |
| **LAPA** | — | VQ-VAE | 离散潜在动作 | ✅ | 无监督 VQ-VAE | ❌ |

### 1.2 关键发现

**发现1：所有人都在用"小视觉编码 + 大 Transformer"**

Octo 的经验最直接——他们刻意把 CNN 做浅（SmallStem16，16×16 patch），让 Transformer 承担尽可能多的计算。ImageNet 预训练权重对机器人任务**没有帮助**。

```
RT-1（老方案）:   ResNet-50(大CNN) → 小Transformer   ← 不要这样
Octo（新方案）:   小CNN → 大Transformer                ← 这是对的
GR-1（新方案）:   ViT-B(冻结) → Perceiver Resampler → GPT-2风格Transformer
π₀（新方案）:     SigLIP ViT(冻结) → MoE双流Transformer
```

**发现2：Perceiver Resampler 是标准做法**

GR-1 和 Seer 都用了 Perceiver Resampler——用少量可学习 query token 从 ViT 的大量 patch token 中压缩信息。这比直接取 CLS token 好得多：

```
ViT patch tokens (197个, 768-d) ──► Perceiver Resampler ──► N个压缩token
                                      (learnable queries)       (如64个, 384-d)
```

**发现3：视频预测是最有效的预训练任务**

GR-1 用 Ego4D（3500小时第一人称视频）做视频预测预训练，仅用 10% 机器人数据微调就达到 77.8% 成功率（CALVIN）。视频预测是一个天然的"世界模型"学习任务。

**发现4：因果 Transformer 是统一架构选择**

GR-1、Octo、Seer、π₀ 全部使用因果 Transformer（GPT 风格），区别仅在于注意力掩码模式：
- GR-1：标准因果掩码
- Octo：逐块掩码（block-wise）
- π₀：块因果掩码（VLM block + Action Expert block）

### 1.3 对 RDAE 的启示

当前 RDAE 编码器的最大问题是：
- ❌ 用了大 CNN（ResNet-50）而不是小 CNN + 大 Transformer
- ❌ 3 个 token 的 Transformer 不是真正的时序建模
- ❌ 没有 Perceiver Resampler——直接取 CLS token 丢掉所有空间信息

**应该改成**（借鉴 GR-1 + Octo + Seer）：

```
RGB (T, 224, 224) ──► 冻结 ViT-B/MAE ──► (T, 197 patch tokens, 768)
                                                    │
                                                    ▼
                                           Perceiver Resampler
                                           (64 learnable queries)
                                                    │
                                                    ▼
                                           (T, 64 tokens, 384)
                                                    │
                         ┌──────────────────────────┘
                         ▼
关节 (T, 14) ──► MLP ──► (T, 1 token, 384)
                         │
                         ▼
            ┌──────────────────────────┐
            │  GPT 风格因果 Transformer │
            │  (8层, 384-d, 12头)       │
            │                          │
            │  Token序列:               │
            │  [IMG_0]...[IMG_T] [STATE_0]...[STATE_T] [CLS] │
            └──────────┬───────────────┘
                       │
                       ▼
                  z_w (T, 256)
```

---

## 2. 动作解码器调研

### 2.1 主流方案对比

| 项目 | 机构 | 方法 | 架构 | 推理速度 | 代码 |
|------|------|------|------|---------|:---:|
| **Diffusion Policy** | Columbia | DDPM/DDIM | ResNet-18 + 1D CNN UNet + FiLM | 10步 DDIM | ✅ |
| **3D Diffuser Actor** | CMU | 3D Diffusion | 3D UNet on point clouds | — | ✅ |
| **Dita** | — | Diffusion Transformer | Transformer denoiser | — | ✅ |
| **DiVLA** | — | AR + Diffusion 混合 | Transformer + Diffusion head | 82Hz | ✅ |
| **π₀** | Physical Intelligence | Flow Matching | MoE Transformer | 50Hz, 50 chunks | ✅ |
| **Falcon** | — | Partial Denoising | 训练无关加速插件 | 2-7× 加速 | ✅ |
| **RoLD** | — | Latent Diffusion | Autoencoder + Latent Diffusion | >2× 加速 | ✅ |

### 2.2 关键发现

**发现1：Diffusion Policy 原始实现用的是 ResNet-18 + FiLM**

```
Diffusion Policy (Chi 2023) 原始架构:
  Visual Encoder: ResNet-18 (无预训练) + GroupNorm + Spatial Softmax Pooling
  Denoising Net: 1D CNN UNet
    - down_dims: [256, 512, 1024]
    - kernel_size: 5
    - FiLM conditioning at every layer
  Sampling: DDIM (10步推理，质量接近100步DDPM)
```

这是 RDAE 当前动作解码器的直接参照。

**发现2：FiLM 对于困难任务是必要的，但对简单任务不必要**

UCSD 的组件分析论文（"Unpacking the Individual Components of Diffusion Policy"）发现：
- UNet 架构对 **PegInsertion、TurnFaucet** 等困难任务至关重要，MLP 对简单任务就够
- FiLM 条件对困难任务提升显著，对简单任务非必要
- RDAE 的目标任务是桌面操作（抓取、放置等），难度中等——UNet + FiLM 是合适的选择

**发现3：DDIM 10步推理是标准做法**

原始 Diffusion Policy 论文使用 DDIM 10步采样而非 DDPM 100步。当前 RDAE 用 DDPM 100步，推理慢 10 倍——这是低挂果实。

**发现4：SOTA 正在往 Transformer 扩散方向走**

Dita 和 DiVLA 都用 Transformer 替代 CNN UNet 做去噪器，利用 Transformer 的注意力机制天然处理长序列依赖。但这需要更大的训练数据和计算量——适合 Stage 2+。

### 2.3 对 RDAE 的启示

当前动作解码器的设计**方向正确**（Diffusion Policy + 1D UNet + FiLM），需要的优化：
- ✅ 架构选择正确——保留 1D CNN UNet
- ⚠️ 从 DDPM 100步 → DDIM 10步（推理加速 10×，0.5天工作量）
- ⚠️ 增加 action horizon 可配置性和 receding horizon control
- ❌ 不需要大改——Diffusion Policy 的组件分析证实 CNN UNet 对此类任务足够

---

## 3. 逆动力学/传感器重建调研

### 3.1 主流方案对比

| 项目 | 机构 | 方法 | 输入 → 输出 | 架构 | 代码 |
|------|------|------|------------|------|:---:|
| **Seer** | 上海 AI Lab | PIDM | 视觉历史+预测 → 动作 | Transformer + [INV] token | ✅ |
| **AMPLIFY** | Georgia Tech | Motion Token IDM | Motion tokens → 动作 | 自回归 Transformer | ❌ |
| **FILIC** | — | 雅可比力估计 | 关节扭矩 → 末端力 | 解析模型 + 数字孪生 | ✅ |
| **LIP4RobotID** | MERL | GP 逆动力学 | 状态 → 关节扭矩 | Gaussian Process | ✅ |
| **经典逆动力学** | 通用 | τ = M(q)q̈ + C(q,q̇) + G(q) | 状态+加速度 → 关节扭矩 | 解析/学习 | — |

### 3.2 关键发现

**发现1：Seer 的 [INV] Token 设计是 RDAE 最直接的参照**

Seer 在因果 Transformer 中引入专门的 `[INV]`（Inverse Dynamics）token，用单向注意力掩码让它能"看到"过去和预测的未来视觉状态，从而预测中间的动作序列。

RDAE 的传感器解码器本质上也是在做一个类似的事情——从世界潜空间 + 预测动作 → 反推传感器读数。Seer 的做法证实了"在 Transformer 中用专用 token 做逆动力学"是可行的。

**发现2：直接回归力/扭矩很困难**

FILIC 和 LIP4RobotID 都采用了**物理先验**（雅可比矩阵、拉格朗日方程）来辅助力估计。纯数据驱动的方法（如当前 RDAE 的 3 层 MLP）在面对接触事件的不连续性时表现很差。

**发现3：传感器估计缺乏专门的 SOTA 开源方案**

搜索发现，**没有专门针对"从视觉潜空间+动作序列重建多模态传感器"的开源项目**。最接近的是：
- FILIC（仅力估计，需要物理模型）
- AMPLIFY（逆模型，但只预测动作不预测传感器）
- Seer（逆动力学动作预测，不是传感器重建）

这意味着 RDAE 的传感器重建模块是一个**相对新颖的设定**，没有现成的开源方案可以直接套用。但也意味着需要更多实验来验证。

### 3.3 对 RDAE 的启示

当前传感器解码器的设计是三个模块中**最弱的**：

| 问题 | 严重性 | 解决方案 |
|------|:---:|------|
| mean_pool 丢失时序 | 🔴 | 用 Transformer/GRU 编码动作序列 |
| MLP 无法建模接触 | 🔴 | 引入接触检测头（二分类）+ 分场景预测 |
| 无物理约束 | 🟠 | 加入拉格朗日动力学损失（LIP4RobotID 风格） |
| 无现成方案可抄 | 🟡 | 参照 Seer 的 [INV] token + FILIC 的物理约束 |

---

## 4. 当前设计差距分析

### 4.1 世界编码器：差距最大

| 维度 | 当前 RDAE | SOTA（GR-1 / Octo / Seer） | 差距 |
|------|----------|---------------------------|:---:|
| 视觉编码器 | ResNet-50 (23M, ImageNet预训练) | ViT-B/MAE (冻结) + Perceiver Resampler | 🔴 |
| 融合方式 | 2层 Transformer, 3 token | 8-12层因果 Transformer, 数百 token | 🔴 |
| 时序建模 | 外部 frame_stack | 因果注意力 + 时序位置编码 | 🔴 |
| 预训练 | ImageNet 分类 | 视频预测 (Ego4D) / VLM | 🔴 |
| 潜空间约束 | 无 | 视频重建 Loss / 对比 Loss | 🟠 |
| 参数量 | ~28M | ~46M-200M（可训练部分） | 🟢 |

### 4.2 动作解码器：差距最小

| 维度 | 当前 RDAE | Diffusion Policy (Chi) | 差距 |
|------|----------|----------------------|:---:|
| 方法 | Diffusion (DDPM 100步) | Diffusion (DDIM 10步) | 🟡 |
| 架构 | 1D CNN UNet + FiLM | 1D CNN UNet + FiLM | 🟢 |
| 条件机制 | FiLM | FiLM | 🟢 |
| 视觉编码 | 外部编码器 | ResNet-18 + GroupNorm | 🟡 |
| 推理速度 | ~0.5s (100步) | ~0.01s (10步DDIM) | 🟡 |

### 4.3 传感器解码器：差距最大 + 无直接参照

| 维度 | 当前 RDAE | 最接近的 SOTA（Seer/AMPLIFY） | 差距 |
|------|----------|------------------------------|:---:|
| 时序处理 | mean_pool → 单向量 | 因果 Transformer 逐帧处理 | 🔴 |
| 架构 | 3层 MLP | Transformer + 专用 token | 🔴 |
| 物理约束 | 无 | 拉格朗日损失（LIP4RobotID） | 🔴 |
| 接触建模 | 无 | 分类头 + 回归头分离 | 🔴 |
| 多模态输出 | 3个线性头 | 多任务 Transformer 解码 | 🟠 |
| 开源参照 | — | 无直接对应的开源项目 | 🟡 |

---

## 5. 修正后的架构建议

### 总体架构（对标 GR-1 + Seer + Diffusion Policy）

```
                        ┌─────────────────────┐
                        │   冻结 ViT-B/MAE     │
                        │   (视觉编码器)       │
                        └──────────┬──────────┘
                                   │ patch tokens (T, 197, 768)
                                   ▼
                        ┌─────────────────────┐
                        │  Perceiver Resampler │  ← GR-1 / Seer 标准做法
                        │  (64 query tokens)   │
                        └──────────┬──────────┘
                                   │ compressed (T, 64, 384)
                                   ▼
┌──────────┐    ┌─────────────────────────────────────┐
│ 关节状态  │───►│        因果 Transformer              │ ← GPT-2 风格
│ (T, 14)  │    │  (8层, 384-d, 12头)                  │   参照 GR-1 / Octo / Seer
└──────────┘    │                                      │
                │  Token: [IMG_tokens | STATE_tokens]  │
                │  输出: [CLS] → z_w (256)              │
                │        [FRS] → 未来帧预测 (辅助任务)   │ ← Seer 的前瞻设计
                │        [INV] → 传感器预测入口          │ ← Seer 的逆动力学设计
                └──────────┬──────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌───────────┐ ┌────────────┐
    │  z_w (256)  │ │ [FRS]预测 │ │ [INV]特征   │
    │  世界潜空间  │ │  视频帧    │ │  逆动力学   │
    └──────┬──────┘ └───────────┘ └──────┬─────┘
           │                             │
           ▼                             ▼
┌─────────────────────┐    ┌─────────────────────────┐
│   动作解码器          │    │    传感器解码器           │
│   (Diffusion Policy) │    │                         │
│                      │    │  [INV]特征 + z_w + 动作  │
│   1D CNN UNet + FiLM │    │     ▼                   │
│   DDIM 10步采样      │    │  Temporal Transformer   │
│   输出: 动作序列      │    │  (2层, 384-d)            │
│   (B, T, 7)          │    │     ▼                   │
│                      │    │  多头输出:               │
│                      │    │  · joint_pos (T, 7)     │
│                      │    │  · joint_vel (T, 7)     │
│                      │    │  · force (T, 6)          │
│                      │    │  · contact (T, 1) ← 二分类│
└─────────────────────┘    └─────────────────────────┘
```

### 训练策略（对标 GR-1 两阶段）

```
阶段A（预训练，使用动作自由视频）:
  - 冻结 ViT + Perceiver Resampler
  - 损失: 视频帧预测 MSE（[FRS] token 重建未来帧）
  - 数据: Ego4D / Something-Something / YouTube 片段
  - 目标: 让编码器学会场景动力学

阶段B（微调，使用机器人数据）:
  - 解冻全部参数
  - 损失: λ₁·action_MSE + λ₂·joint_MSE + λ₃·force_MAE + λ₄·contact_BCE + λ₅·video_MSE
  - 数据: BridgeData V2 / DROID
  - 目标: 端到端学会视频→动作→传感器

阶段C（仿真验证）:
  - RDAE Pipeline: 视频 → 编码器 → 动作+传感器 → MuJoCo → 一致性过滤
```

### 参数量估算

| 模块 | 参数 | 可训练 |
|------|------|:---:|
| ViT-B/MAE（冻结） | 86M | ❌ |
| Perceiver Resampler | ~2M | ✅ |
| 因果 Transformer (8层) | ~28M | ✅ |
| 动作解码器 (1D UNet) | ~1.5M | ✅ |
| 传感器解码器 (小Transformer) | ~3M | ✅ |
| **总计** | **~120M** | **~35M** |

---

## 6. HuggingFace 关键发现

> 来自 HuggingFace Models + Papers + LeRobot 的补充调研

### 6.1 最相关的项目（2024-2025）

| 项目 | 机构 | 核心思路 | RDAE 关联度 | HuggingFace |
|------|------|---------|:---:|------|
| **FLARE** | — | 扩散 Transformer 内部对齐未来观测的潜嵌入，支持人类视频协同训练 | ⭐⭐⭐ | `papers` |
| **MCR** | — | DROID 上预训练视觉编码器，对比损失对齐视觉观测与本体状态-动作动力学 | ⭐⭐⭐ | `papers/2410.22325` |
| **RoboBERT** | — | CALVIN SOTA (4.52 ABCD-D)，轻量 CNN+Diffusion，无辅助数据 | ⭐⭐ | — |
| **RDT-1B** | — | 1B 扩散 Transformer，46 数据集，SigLIP+T5，64 动作块 | ⭐⭐ | `robotics-diffusion-transformer/rdt-1b` |
| **GR00T-N1** | NVIDIA | 跨具身基础模型，SigLip2+T5+Flow Matching，2B | ⭐⭐ | `nvidia/GR00T-N1-2B` |
| **Cosmos** | NVIDIA | 世界基础模型，30秒视频生成，多视角，物理推理 | ⭐ | `blog` |
| **LeRobot** | HuggingFace | 开源机器人学习库，预置 Diffusion Policy checkpoint | ⭐⭐⭐ | `lerobot/diffusion_pusht` |

### 6.2 FLARE — RDAE 最直接的参照之一

```
FLARE 架构核心:
  Diffusion Transformer 特征 ←→ 未来观测潜嵌入
                                    │
  人类视频（无动作标签）───────────► 协同训练 ──► 泛化提升
  极少机器人演示（甚至1条）─────────► 微调
```

与 RDAE 的相似之处：
- ✅ 都从视频学习世界动态
- ✅ 都利用潜空间（FLARE 用 Diffusion Transformer 的潜特征）
- ✅ 都支持人类视频+机器人数据联合训练
- 不同：FLARE 是策略模型（输出动作），RDAE 是数据引擎（输出合成数据）

### 6.3 MCR — 视觉预训练的最佳参照

MCR（Manipulation Centric Representation）的核心做法：

```
DROID 数据集 (350小时机器人数据)
        │
        ▼
┌───────────────────────────────────┐
│  三重损失联合训练:                   │
│  · 对比损失: 视觉 ↔ 本体状态-动作    │
│  · 行为克隆损失: 视觉 → 动作         │
│  · 时序对比损失: t帧 ↔ t+Δt帧       │
└───────────────────────────────────┘
        │
        ▼
  视觉编码器（ResNet-50, ~26M）
  提升: +14.8% (仿真), +76.9% (真实数据效率)
```

🎯 **发现3**：MCR 证实了**"在 DROID 级别的机器人数据上用对比学习 + 行为克隆 + 时序对比联合预训练"**优于 ImageNet 预训练。这直接支持 RDAE 的 Stage 1 策略。

### 6.4 LeRobot — 开箱即用的 Diffusion Policy

HuggingFace 的 LeRobot 库提供了：
- `lerobot/diffusion_pusht` — 预训练 Diffusion Policy（PushT 任务，63.8% 成功率）
- `lerobot/diffusion_pusht_keypoints` — 关键点版本（71.0% 成功率）
- 完整训练+评估代码，可作为 RDAE 动作解码器的参照实现

---

## 7. PapersWithCode 关键发现

### 7.1 扩散策略 SOTA 排行榜

| 排名 | 方法 | CALVIN ABCD→D | 架构 | 亮点 |
|:---:|------|:---:|------|------|
| 1 | **RoboBERT** | 4.52 | CNN + Diffusion | 轻量，无辅助数据 |
| 2 | **Seer** | 4.28 | Transformer + [FRS][INV] | ICLR 2025 Oral |
| 3 | **GR-1** | 4.21 | MAE + 因果Transformer | 视频预训练 |
| 4 | **RoboDual** | 4.04 | VLA + 扩散专家 | 仅 20M 可训练参数 |

### 7.2 值得关注的新范式

| 项目 | 范式 | 核心创新 |
|------|------|---------|
| **HybridVLA** | AR + Diffusion 融合 | 在同一个 LLM 中同时做自回归推理和扩散动作生成 |
| **RoboDual** | 双系统协作 | 通用 VLA（慢思考）+ 轻量扩散专家（快执行），仅需 5% 演示数据 |
| **FP3** | 3D 基础策略 | 首个基于点云的 3D 基础策略，80条演示即达 90% 成功率 |
| **Dreamitate** | 视频生成即策略 | 微调视频扩散模型，生成的视频直接控制机器人（绕过 embodiment gap） |

---

## 8. 可用的开源代码库与模型

| 优先级 | 资源 | 类型 | 用途 |
|:---:|------|:---:|------|
| ⭐⭐⭐ | [GR-1](https://github.com/bytedance/GR-1) | GitHub | 编码器架构参照（MAE+Perceiver+因果Transformer）|
| ⭐⭐⭐ | [Octo](https://github.com/octo-models/octo) | GitHub | 模块化 Transformer + Diffusion Head |
| ⭐⭐⭐ | [Seer](https://github.com/InternRobotics/Seer) | GitHub | [FRS]+[INV] token + 端到端视觉-动作闭环 |
| ⭐⭐⭐ | [FLARE](https://paperswithcode.com/paper/flare-robot-learning-with-implicit-world) | 论文 | 扩散 Transformer 潜空间对齐未来观测 + 人类视频协同训练 |
| ⭐⭐⭐ | [MCR](https://huggingface.co/papers/2410.22325) | 论文+HF | DROID 对比预训练视觉编码器，+76.9% 数据效率 |
| ⭐⭐⭐ | [LeRobot](https://github.com/huggingface/lerobot) | GitHub+HF | 开源机器人学习库，含预训练 Diffusion Policy |
| ⭐⭐⭐ | [RDT-1B](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | HF Model | 1B 扩散 Transformer，46 数据集，SigLIP+T5 |
| ⭐⭐ | [π₀ openpi](https://github.com/Physical-Intelligence/openpi) | GitHub | VLM + Action Expert + Flow Matching |
| ⭐⭐ | [RoboDual](https://paperswithcode.com/paper/towards-synergistic-generalized-and-efficient) | 论文 | VLA + 轻量扩散专家，仅需 5% 演示数据 |
| ⭐⭐ | [HybridVLA](https://paperswithcode.com/paper/hybridvla-collaborative-diffusion-and) | 论文 | AR + Diffusion 在同一 LLM 中融合 |
| ⭐⭐ | [FP3](https://paperswithcode.com/paper/fp3-a-3d-foundation-policy-for-robotic) | 论文 | 点云 3D 基础策略，80条演示→90%成功率 |
| ⭐ | [LIP4RobotID](https://github.com/merlresearch/LIP4RobotInverseDynamics) | GitHub | 物理约束逆动力学（GP回归）|
| ⭐ | [FILIC](https://github.com/TATP-233/FILIC) | GitHub | 力估计（关节扭矩→末端力）|

---

## 9. 参考文献

1. **GR-1**: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation" — ByteDance, ICLR 2024. [GitHub](https://github.com/bytedance/GR-1)
2. **Octo**: "Octo: An Open-Source Generalist Robot Policy" — Octo Model Team, RSS 2024. [GitHub](https://github.com/octo-models/octo) | [HF](https://huggingface.co/rail-berkeley/octo-small-1.5)
3. **Seer**: "Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation" — Shanghai AI Lab, ICLR 2025 Oral. [GitHub](https://github.com/InternRobotics/Seer)
4. **π₀**: "π₀: A Vision-Language-Action Flow Model for General Robot Control" — Physical Intelligence, 2024. [GitHub](https://github.com/Physical-Intelligence/openpi) | [HF Blog](https://huggingface.co/blog/pi0)
5. **AMPLIFY**: "Actionless Motion Priors for Robot Learning from Videos" — Georgia Tech, 2025. [arXiv](https://arxiv.org/abs/2506.14198)
6. **Diffusion Policy**: "Visuomotor Policy Learning via Action Diffusion" — Chi et al., RSS 2023. [LeRobot HF](https://huggingface.co/lerobot/diffusion_pusht)
7. **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation" — Meta, CoRL 2022. [GitHub](https://github.com/facebookresearch/r3m)
8. **Unpacking DP**: "Unpacking the Individual Components of Diffusion Policy" — Xiu Yuan, UCSD, 2024. [arXiv](https://arxiv.org/abs/2412.00084)
9. **MCR**: "Robots Pre-train Robots: Manipulation-Centric Robotic Representation" — 2024. [HF Paper](https://huggingface.co/papers/2410.22325)
10. **FLARE**: "Robot Learning with Implicit World Modeling" — 2025. [PapersWithCode](https://paperswithcode.com/paper/flare-robot-learning-with-implicit-world)
11. **RDT-1B**: "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation" — 2024. [HF Model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)
12. **RoboBERT**: "End-to-End Multimodal Manipulation" — 2025, CALVIN SOTA. [PapersWithCode](https://paperswithcode.com)
13. **HybridVLA**: "Collaborative Diffusion and Autoregression in a Unified VLA Model" — 2025. [PapersWithCode](https://paperswithcode.com/paper/hybridvla-collaborative-diffusion-and)
14. **RoboDual**: "Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation" — 2025. [PapersWithCode](https://paperswithcode.com/paper/towards-synergistic-generalized-and-efficient)
15. **FP3**: "A 3D Foundation Policy for Robotic Manipulation" — 2025. [PapersWithCode](https://paperswithcode.com/paper/fp3-a-3d-foundation-policy-for-robotic)
16. **GR00T-N1**: "NVIDIA GR00T-N1-2B" — NVIDIA, 2025. [HF Model](https://huggingface.co/nvidia/GR00T-N1-2B)
17. **NVIDIA Cosmos**: "World Foundation Models for Physical AI" — NVIDIA, 2025. [HF Blog](https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5)
18. **LeRobot**: HuggingFace Robotics Library. [GitHub](https://github.com/huggingface/lerobot)
