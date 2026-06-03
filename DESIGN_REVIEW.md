# RDAE Stage-1 设计评审

> 2026-06-03 | 针对三大核心模块的设计分析、风险评级和优化建议

---

## 总览

| 模块 | 当前设计 | 参照来源 | 风险 | 是否需要现在优化 |
|------|---------|---------|------|:---:|
| **世界编码器** | ResNet + 2层 Transformer（3 token融合） | AMPLIFY, ViPRA, LAPA | 🔴 高 | **是，阻塞性问题** |
| **动作解码器** | 1D Conv UNet + FiLM，100步DDPM | Diffusion Policy (Chi 2023) | 🟡 中 | 可后续迭代 |
| **传感器解码器** | 3层 MLP，mean-pool 动作 | AMPLIFY inverse model | 🔴 高 | **是，精度瓶颈** |

---

## 1. 多模态世界编码器 — 🔴 高优先级

### 参照来源

| 方法 | 核心思路 | 与当前设计的差距 |
|------|---------|-----------------|
| **AMPLIFY** (Collins 2024) | 分离前向运动模型（从视频学运动先验）和逆模型（从少量带动作数据学控制） | 当前没有运动先验预训练 |
| **ViPRA** (Routray 2025) | 视频预测模型 + 光流一致性校正，学习潜在连续动作 | 没有光流约束，潜空间可能没有物理意义 |
| **LAPA** (Ye 2025) | VQ-VAE 量化帧间运动为离散潜在动作 | 没有量化机制，潜空间无结构化 |
| **DINOv2** | 自监督视觉预训练 | 用了 ImageNet 预训练，但没做机器人域适应 |

### 核心问题

```
当前架构:
  RGB (224×224×3) ──► ResNet ──► 1 个 512-d 向量 ──┐
                                                     ├──► [CLS][VIS][STATE] 3个token
  关节 (14,) ────────► MLP ────► 1 个 512-d 向量 ──┘      │
                                                           ▼
                                                    2层 Transformer
                                                           │
                                                           ▼
                                                       z_w (256)
```

**问题1：空间信息被完全压缩**。一张 224×224 的 RGB 图被压缩为单个 512-d 向量，丢失了所有空间位置信息。无法区分"杯子在桌上"和"杯子在手里"的细微空间差异。

**问题2：3 个 token 的 Transformer 等价于简单交叉注意力**。`[CLS, VIS, STATE]` 只有 3 个 token，2 层 Transformer 的计算量微不足道，本质上只是一个加权求和，不是真正的"融合"。

**问题3：没有时序建模**。当前 time 维度靠外部 frame_stack（堆叠 2 帧 → 6 通道输入），但 ResNet 的第一层卷积会把 6 通道直接混合，无法显式建模帧间变化。机器人操作的关键信息（速度、加速度、接触事件）都藏在帧间变化中。

**问题4：潜空间没有结构化约束**。没有重建损失、对比损失或量化约束，潜空间可能坍缩到平凡解。ViPRA 用光流一致性保证潜空间有物理意义，LAPA 用 VQ-VAE 保证离散结构化——当前设计两者都没有。

### 优化方案

**方案A（轻量修复，本周可完成）— 推荐先做**

```
RGB (T, C, H, W) ──► ViT (保留 patch 特征) ──► (T, N_patches+1, 768)
                                                       │
                                                       ├──► Cross-Attention
                                                       │    (Visual 作为 K,V)
关节 (T, 14) ────► MLP + PosEmb ──────────────────────► (T, 768) 作为 Q
                                                       │
                                                       ▼
                                                 Temporal Transformer
                                                 (因果注意力, 4层)
                                                       │
                                                       ▼
                                                 z_w (T, 256)
```

改进点：
1. **保留 ViT patch 特征**：不再压缩为单向量，保留 197 个 patch token（含 CLS），保留空间信息
2. **Cross-Attention 融合**：用关节状态作为 Query 查询视觉特征中的相关区域
3. **Temporal Transformer**：4 层因果注意力显式建模帧间动态
4. **增加重建辅助损失**：用 z_w 预测下一帧的关节状态作为自监督信号

**方案B（对标 SOTA，Stage 2 做）**
- VQ-VAE 潜空间量化（LAPA 风格）
- 光流一致性损失（ViPRA 风格）
- 对比学习：同场景不同视角的 z_w 应该相近

---

## 2. 动作解码器 — 🟡 中优先级

### 参照来源

| 方法 | 核心思路 | 与当前设计的差距 |
|------|---------|-----------------|
| **Diffusion Policy** (Chi 2023) | 1D UNet + DDIM，在动作序列上扩散 | 当前用 DDPM 而非 DDIM，推理慢 10× |
| **π₀** (Physical Intelligence 2024) | 大规模 Transformer 扩散，多任务 | 架构相似但规模差距大 |

### 核心问题

**问题1：DDPM 采样太慢**。100 步 DDPM 每步都要过 UNet，生成一条 16 步动作序列需要 100 次前向传播。Chi 等人的原始论文用 **DDIM**（去噪扩散隐式模型），只需 10 步采样质量就接近 100 步 DDPM。

**问题2：FiLM 条件机制偏弱**。当前用 scale/shift 逐层调制，但条件信号只有 256-d z_w + time_emb，不能选择性关注动作序列的不同部分。更好的做法是用 Cross-Attention：让 UNet 中间特征 attend to z_w。

**问题3：UNet 的感受野固定为 kernel_size=5**。对于 16 步动作序列，kernel_size=5 意味着每层感受野约 5 步（3 层编码器叠加约 13 步），勉强覆盖。但如果后续增加到 32 步预测，就不够了。

### 优化方案

**方案A（轻量修复）— 推荐先做**

```python
# 1. DDIM 采样替代 DDPM（推理加速 10×）
def _sample_ddim(self, z_w, steps=10):
    # 100 → 10 步，质量几乎无损

# 2. Cross-Attention 替代 FiLM
class CrossAttnUNet1D(nn.Module):
    # bottleneck 处加一层 Cross-Attention
    # z_w 作为 K,V，UNet 特征作为 Q

# 3. 可变 kernel_size 和 dilation
# 低层用大 kernel (7→5→3)，扩张卷积覆盖长距依赖
```

这些改动量不大但效果显著，可以在 Stage-1 后期做。

---

## 3. 传感器解码器 — 🔴 高优先级

### 参照来源

| 方法 | 核心思路 | 与当前设计的差距 |
|------|---------|-----------------|
| **AMPLIFY** inverse model | 从视频特征+动作预测本体感觉 | 用的是序列模型，不是单帧 MLP |
| **Inverse Dynamics** (经典) | f(s_t, s_{t+1}) → a_t，用于好奇驱动探索 | 反向：已知状态变化→推测传感器 |

### 核心问题

```
当前:
  z_w (256) ⊕ mean_pool(actions over 16 steps) (7) ──► MLP ──► joint/force

问题: mean_pool 把 16 步动作压缩成一个向量，完全丢失时序！
```

**这是最薄弱的一环**。力传感器的读数取决于运动的瞬时加速度和接触状态，不能从"平均动作"推断。举例：机械臂缓慢靠近桌面和快速撞击桌面，平均动作可能相同，但力传感器读数天差地别。

**问题1：丢失时序**。`actions.mean(dim=-1)` 把 16 步动作压成 1 个向量。

**问题2：无法建模接触事件**。接触力是**不连续**的（碰到物体瞬间从 0 跳变），普通 MLP 很难学习这种不连续性。

**问题3：输入信息不足**。仅从 z_w + 动作预测力，缺少物体物理属性（质量、刚度、摩擦系数）。白皮书 §70 提到需要 Affordance/Dynamics/Material Decoders 提供这些——但目前完全没有。

### 优化方案

**方案A（必须修）— 推荐立刻做**

```python
class SensorDecoder(nn.Module):
    """
    改进版：序列建模 + 物理先验
    """
    def __init__(self):
        # 1. 动作序列编码器（Transformer / GRU）
        self.action_encoder = nn.GRU(7, 256, num_layers=2, batch_first=True)
        
        # 2. 时序传感器解码器
        # 输入: [z_w (每帧), action_encoding (每帧)]
        # 输出: 逐帧的 joint_pos, joint_vel, force
        self.temporal_decoder = nn.TransformerDecoder(...)
        
    def forward(self, z_w, actions):
        """
        z_w: (B, T, 256)  — 现在有 T 维了！
        actions: (B, T, 7) — 每步动作，不再 mean_pool
        """
        # 编码动作序列
        action_feat, _ = self.action_encoder(actions)  # (B, T, 256)
        
        # Cross-attention: 每帧用世界状态 attend 动作特征
        combined = torch.cat([z_w, action_feat], dim=-1)  # (B, T, 512)
        
        # 逐帧预测（用 Transformer 建模帧间依赖）
        outputs = self.temporal_decoder(combined)
        
        # 多头输出
        joint_pos = self.joint_pos_head(outputs)  # (B, T, 7)
        joint_vel = self.joint_vel_head(outputs)  # (B, T, 7)  
        force = self.force_head(outputs)          # (B, T, 6)
        
        # 🔑 新增：物理一致性损失
        # 预测的力和关节加速度应满足拉格朗日动力学
        # M(q) * q̈ + C(q,q̇) + G(q) = τ + J^T * F_ext
        return {...}
```

**方案B（Stage 2）**
- 加入白皮书 §70 的物理属性解码器（质量、摩擦系数等）
- 用拉格朗日/牛顿方程作为物理约束损失

---

## 优先级排序

| 序号 | 改动 | 影响 | 工作量 | 阻塞后续？ |
|:----:|------|------|:------:|:---:|
| **1** | 世界编码器：保留 patch 特征 + Temporal Transformer | 提高表示质量 | 2-3天 | ✅ 是 |
| **2** | 传感器解码器：序列建模替代 mean_pool | 力/关节预测可用 | 1-2天 | ✅ 是 |
| **3** | 世界编码器：增加重建/对比辅助损失 | 防止潜空间坍缩 | 1天 | ⚠️ 半阻塞 |
| **4** | 动作解码器：DDIM 采样 | 推理加速 10× | 0.5天 | ❌ 否 |
| **5** | 动作解码器：Cross-Attention 替代 FiLM | 小幅提升精度 | 1天 | ❌ 否 |

## 建议执行顺序

```
第一轮（本周，阻塞性修复）:
  1. 世界编码器 → patch保留 + Temporal Transformer
  2. 传感器解码器 → 序列建模

第二轮（下周，质量提升）:
  3. 辅助损失（重建 + 对比）
  4. DDIM 采样

第三轮（Stage 2，迭代）:
  5. VQ-VAE 量化
  6. 光流一致性
  7. 物理属性解码器
```

---

## 结论

你对地基的担忧是对的。当前设计有两个 **🔴 阻塞性问题**：

1. **世界编码器**：3 token 的 Transformer 不是真正的融合，单向量压缩丢失了所有空间信息
2. **传感器解码器**：mean_pool 动作 → MLP 无法建模力和接触，预测基本不可用

建议**现在就修**这两块，大约 3-5 天工作量。修好之后端到端训练才能产出可用的合成数据。动作解码器可以先跑通，DDIM 优化不急。
