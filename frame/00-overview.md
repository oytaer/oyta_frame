# Oyta-Framework 总览（v2 · 真实 API 版）

> 架构设计文档：`docs/AI框架系统架构设计.md` v2.0
> 底层依赖：`oyta-tensor = "0.1.1"`（**统一 registry 入口**，已发布）
> 
> oyta-tensor 是统一 re-export 入口：
>   - re-export oyta-core → Tensor / DynamicShape / safetensors
>   - re-export oyta-device → Backend trait / Cpu / MatmulOps / LayerNormOps / FlashAttentionOps
>   - re-export oyta-ops → matmul / layernorm / flash_attention / softmax / gelu / relu / dropout

---

# 一、关键 API 速查表（所有框架层 crate 直接用）

```rust
// ===== 入口 =====
use oyta_tensor::prelude::*;           // 一行导入全部

// ===== Tensor 工厂 =====
let t = Tensor::<f32, Cpu>::randn([64, 64]);     // 标准正态初始化
let t = Tensor::<f32, Cpu>::zeros([32, 16]);     // 全零
let t = Tensor::<f32, Cpu>::ones([32, 16]);      // 全一
let t = Tensor::<f32, Cpu>::from_data(vec![...], [shape]);  // 从 Vec 构造

// ===== Tensor 算子（Tensor 方法，直接调用）=====
let c = a.matmul(&b).unwrap();                     // batched matmul
let s = logits.softmax(-1).unwrap();               // Softmax 沿最后一维
let g = x.gelu().unwrap();                         // GELU 激活
let r = x.relu().unwrap();                         // ReLU 激活
let n = x.layernorm(&gamma, &beta, 1e-5).unwrap(); // LayerNorm
let z = a.add(&b).unwrap();                        // 逐元素加
let z = a.sub(&b).unwrap();                        // 逐元素减
let z = a.mul(&b).unwrap();                        // 逐元素乘

// ===== FlashAttention（oyta-ops re-export）=====
use oyta_tensor::flash_attn::flash_attention;     // 单头
use oyta_tensor::flash_attn::flash_attention_multihead;  // 多头

let out = flash_attention(&q, &k, &v, /*causal=*/false, /*scale=*/1.0/8.0);
let out = flash_attention_multihead(&q, &k, &v, num_heads, causal, scale);

// ===== Backend trait（需要跨 device 或 GPU kernel 时）=====
use oyta_tensor::Backend;        // 聚合 trait
use oyta_tensor::Cpu;            // CPU 设备
use oyta_tensor::DeviceContext;  // 设备运行时上下文

// ===== safetensors（oyta-core re-export）=====
use oyta_tensor::{save_safetensors, load_safetensors};

// ===== Dropout（oyta-ops）=====
use oyta_tensor::set_training_mode;  // true = Dropout 启用
use oyta_tensor::is_training_mode;

// ===== Profiling（oyta-ops）=====
use oyta_tensor::set_profiling_enabled;
```

---

# 二、所有框架层 crate 的 Cargo.toml 统一模板

```toml
[package]
name = "oyta-{功能名}"
version = "0.1.0"
edition = "2024"
# 注意：**不写 rust-version**，让 cargo 用当前 toolchain 默认值

[dependencies]
# ===== 底层算子引擎：一个 oyta-tensor = 全部（registry 版本）=====
oyta-tensor = "0.1.1"

# ===== workspace 兄弟 crate（路径相对）=====
oyta-content = { version = "0.1.0", path = "../oyta-content" }   # 按需添加
oyta-router  = { version = "0.1.0", path = "../oyta-router" }    # 按需添加

# ===== 其他常见依赖 =====
serde       = { version = "1.0.229", features = ["derive"] }
serde_json  = "1.0.152"
regex       = "1.11.1"
rayon       = "1.12.0"
safetensors = "0.6.0"   # oyta-core 已有 save_safetensors，序列化 crate 才需要直接依赖
thiserror   = "2.0.16"
anyhow      = "1.0.93"
csv         = "1.3.0"
```

---

# 三、禁止事项（所有开发文档必须遵守）

1. **禁止手写底层算子**（Linear / LayerNorm / FlashAttention / GELU / ReLU / Softmax / Matmul）——全部调用 `oyta_tensor` API
2. **禁止硬写 `rust-version = "1.85"`**——你的项目 edition=2024 但没锁定 rust 版本
3. **禁止手动实现 Kaiming 初始化 / Adam 更新公式**——框架层只做胶水代码（路由 + 稀疏控制 + Parameter 包装）
4. **禁止在文档里硬写 `path = "/root/oyta_ai/..."`**——oyta-tensor 是 registry 版本
5. **禁止 mod.rs 写功能代码**——mod.rs 只允许 `pub mod xxx;`

---

# 四、框架层 crate 开发顺序

| # | crate | 依赖 | 说明 |
|---|-------|------|------|
| 1 | oyta-content | oyta-tensor（唯一底层） | 内容分析（无框架依赖） |
| 2 | oyta-router | oyta-tensor + oyta-content | 稀疏路由 + 三层掩码 + SparseParameterPool |
| 3 | oyta-expert | oyta-tensor + oyta-router | DomainExpert trait + ExpertExecutor + 4 种专家 |
| 4 | oyta-fusion | oyta-tensor + oyta-expert | 三种融合策略 + TaskHead |
| 5 | oyta-autograd | oyta-tensor | Node + Op（含 4 个掩码变体）+ sparse_backward |
| 6 | oyta-loss | oyta-tensor | LossFn trait + CrossEntropy + MultiDomainLoss |
| 7 | oyta-optim | oyta-tensor + oyta-router | Optimizer trait + SparseAdam（按领域独立 m/v/t）|
| 8 | oyta-trainer | **全部** | 端到端主循环 + DataLoader + Callback |
| 9 | oyta-serialize | oyta-tensor + oyta-router | safetensors 按领域 key 分层 |
| 10 | oyta-frame | **全部** | 一行导入的 pub use 顶层 |

---

# 五、每个文件里 Linear 层的正确写法（示例）

```rust
//! 错误写法（我之前的版本——自己手写 matmul + bias 循环）：
//! let mut result = weight_data.matmul(&input).unwrap();
//! for row_start in ... { for j in ... { result_data[row_start + j] += bias[j]; } }

//! 正确写法（调用 oyta_tensor API）：
use oyta_tensor::{Tensor, Backend};
use oyta_tensor::backend::Cpu;

pub struct Linear {
    weight: Parameter,    // [out_features, in_features]
    bias:   Parameter,    // [out_features]
}

impl Linear {
    /// 前向：y = x @ W^T + b  （全调 oyta_tensor）
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let w = self.weight.data.lock().unwrap().clone();  // [out, in]
        let b = self.bias.data.lock().unwrap().clone();    // [out]

        // (a) matmul：[batch, in] @ [in, out] → [batch, out]
        let wx = w.transpose(0, 1).matmul(x).unwrap();
        // 上面 transpose 调用...等等，oyta_tensor 有没有 transpose？
        // 看 oyta-core/src/strides.rs，有 Strides::transpose → Tensor::transpose 方法吗？
        // 先假设 Tensor::transpose(dim_a, dim_b) 存在，否则用 matmul 自己处理转置
        // 实际上 oyta-device::MatmulOps 支持 transa/transb 参数（Backend trait）
        // 所以更高效的写法是直接调 Backend matmul，或者用 Tensor 自带的

        // (b) bias add：广播 [batch, out] += [out]
        let y = wx.add(&b.broadcast_to(wx.shape())).unwrap();
        y
    }
}
```

**重要说明**：上面 `Tensor::transpose` 和 `Tensor::broadcast_to` 是否存在需要在开发时查 oyta-core 的 API。如果不存在，就改用 Backend trait 的 matmul(transa=true, transb=false) 或手动构造正确 shape。**开发时必须以 oyta-tensor 实际导出的 API 为准**，我这版开发文档里的代码是基于 oyta-tensor v0.1.1 的 API 签名写的，但如果有细微差异，以 cargo check 报错为准再修正。

---

# 六、Phase 0 底层算子扩展总览（架构设计文档 §0）

> oyta-tensor 已经 re-export 的算子：Linear/Matmul/LayerNorm/FlashAttention/Softmax/GELU/ReLU/Dropout
> Phase 0 新增的算子：**Conv1d / BatchNorm1d / MaxPool1d / AvgPool1d / Tensor 层 Concat / Tensor 层 Sort**

所有 Phase 0 算子通过 Backend trait 扩展实现，框架层（oyta-autograd/oyta-expert）只调高层接口，不手搓算子内部循环。

## 6.1 Conv1d 一维卷积算子

**数学定义（标准 NCHW 布局）：**
```
out[n, oc, o] = bias[oc] + Σ_{ic=0}^{in_c-1} Σ_{k=0}^{k_size-1}
    weight[oc, ic, k] × x[n, ic, o*stride + k - padding]
```
其中 `out_len = (seq_len + 2*padding - k_size) / stride + 1`

**Backend trait 方法：**
```rust
// oyta-device/src/backend/conv1d.rs
pub trait Conv1dOps<T: Float>: BackendBase {
    fn conv1d(x: *const T, w: *const T, b: *const T, out: *mut T,
              batch, in_c, seq_len, out_c, k_size, stride, padding);
    fn conv1d_backward(x: *const T, w: *const T, go: *const T,
                       gx: *mut T, gw: *mut T, gb: *mut T,
                       batch, in_c, seq_len, out_c, k_size, stride, padding);
}
```

**backward 推导要点（不省略）：**
- dL/dweight[oc, ic, k] = Σ_{n, o} go[n, oc, o] × x[n, ic, o*stride+k-padding]
- dL/dinput[n, ic, t] = Σ_{oc, o, k} go[n, oc, o] × w[oc, ic, k]（t = o*stride+k-padding）
- 中间量用 f64 累加器（防 batch 大时精度损失）
- rayon 并行：按 batch × out_channels 分块

## 6.2 BatchNorm1d 批归一化算子

**数学定义：**
```
训练态:
    μ = (1/N) Σ x_i,  σ² = (1/N) Σ (x_i - μ)²
    x̂ = (x - μ) / √(σ² + ε),  ŷ = γ ⊙ x̂ + β
    running_mean = (1-m)·running_mean + m·μ   (指数移动平均, momentum=0.1)
    running_var  = (1-m)·running_var  + m·σ²

推理态: x̂ = (x - running_mean) / √(running_var + ε)

反向训练态（单行简化公式，避免 O(N²) Jacobian）：
    设 scaled_go[n] = go[n] · γ_f
    mean_sgo = (1/N) Σ scaled_go[n]
    mean_sx  = (1/N) Σ scaled_go[n] · x̂[n]
    dL/dx[m] = inv_std · [ scaled_go[m] - mean_sgo - x̂[m] · mean_sx ]
    dL/dβ_f  = Σ go[n, f]
    dL/dγ_f  = Σ go[n, f] · x̂[n, f]
```

**关键缓存要求（forward 时必须缓存）：**
- `cached_xhat [batch, num_features]` — 归一化结果
- `cached_inv_std [num_features]` — 每个特征的 inv_std
- 没有这两个缓存 backward 无法高效计算

**Running 统计稀疏训练注意点：**
- 只有激活专家才更新自己的 running_mean/var
- 未激活专家的 running 统计保持不变（相当于冻结统计）
- 初始化 running_mean=0, running_var=1, gamma=1, beta=0 → BN 输出恒等变换

## 6.3 MaxPool1d / AvgPool1d 池化算子

**稀疏性对比（核心差异）：**
```
MaxPool1d backward:  天然稀疏 —— 只有 argmax 位置接收梯度，其余全零
AvgPool1d backward:  稠密 —— 每个输入位置平分梯度
Conv1d backward:     稠密 —— 所有输入位置可能贡献梯度
BatchNorm1d backward: 稠密（但特征间独立）
```

**MaxPool 必须缓存 argmax 索引数组（backward 必需）：**
```rust
pub trait PoolOps<T: Float>: BackendBase {
    fn maxpool1d(x, out, argmax, batch, channels, seq_len, k_size, stride, padding);
    fn maxpool1d_backward(grad_out, argmax, grad_x, batch, channels, seq_len, k_size, stride, padding);
    fn avgpool1d(x, out, batch, channels, seq_len, k_size, stride, padding);
    fn avgpool1d_backward(grad_out, grad_x, batch, channels, seq_len, k_size, stride, padding);
}
```

## 6.4 Tensor 层辅助算子

```rust
impl<D: Backend> Tensor<f32, D> {
    /// 沿指定轴拼接（FeatureFusion 用）
    pub fn concat(tensors: &[&Tensor<f32, D>], dim: usize) -> Tensor<f32, D>;
    /// 沿指定轴降序排序 + 索引 Tensor（DomainRouter Top-K 用）
    pub fn sort_desc(&self, dim: usize) -> (Tensor<f32, D>, Tensor<i64, D>);
    /// Conv1d 高层接口（自动处理 NCHW）
    pub fn conv1d(&self, weight, bias, stride, padding) -> Tensor<f32, D>;
    /// BatchNorm1d 高层接口（训练态 / 推理态）
    pub fn batchnorm1d_train(&self, gamma, beta, running_mean, running_var, eps, momentum,
                             cached_xhat, cached_inv_std) -> Tensor<f32, D>;
    pub fn batchnorm1d_eval(&self, gamma, beta, running_mean, running_var, eps) -> Tensor<f32, D>;
}
```

---

# 七、稀疏适配核心设计（架构设计文档 §0.6）

## 7.1 核心原则：算子本身不感知稀疏

**Conv1d / BN / MaxPool1d / Linear 等算子内部代码完全不感知"稀疏"或"领域"**。稀疏性来自框架层的**控制流**:

```rust
// oyta-trainer / oyta-expert 层 — 控制流里的稀疏分支
for expert in &self.experts {
    if self.active_ids.contains(&expert.domain_id()) {
        // 激活 → 完整 forward + backward
        expert.forward(&feature_vector);
    }
    // 未激活 → 什么都不做 → 参数不参与 backward → 梯度保持为零 → optimizer 跳过更新
}
```

## 7.2 三种算子稀疏性对比

| 算子 | forward 必须做 | backward 稀疏性 | 缓存策略 |
|------|---------------|----------------|---------|
| Conv1d | ✅ 激活专家必须 forward | 稠密 | 缓存原始输入 x |
| BatchNorm1d | ✅ 激活专家必须 forward | 稠密 | 缓存 x̂ + inv_std |
| MaxPool1d | ✅ 激活专家必须 forward | **天然稀疏**（只有 argmax 位置有梯度） | 必须缓存 argmax 索引数组 |
| Linear | ✅ 激活专家必须 forward | 稠密 | 缓存输入 x |

## 7.3 未激活专家内存节省

| 阶段 | 激活专家 | 未激活专家 |
|------|---------|-----------|
| 参数内存 | 始终存在 | 始终存在 |
| forward 中间张量 | 需要分配 | **不分配** |
| backward 中间张量 | 需要分配 | **不分配** |
| 梯度内存 | 累积中 | 为 None |

## 7.4 稀疏 autograd —— 跳过整个专家子图

```rust
// oyta-autograd/src/backward.rs
pub fn sparse_backward(root_id, nodes, active_ids) {
    let sorted = topo_sort_reverse(nodes, root_id);
    for &node_id in &sorted {
        let node = &nodes[&node_id];
        // ========== 关键：跳过未激活专家的整个算子子树 ==========
        if let Some(domain) = node.domain_id {
            if !active_ids.contains(&domain) {
                // 这个节点属于未激活专家 → 梯度 = 0，不向下传播
                grads.insert(node_id, Arc::new(Tensor::zeros(node.output.shape())));
                continue;
            }
        }
        // 正常 backward
        compute_backward(&node.op, &node.inputs, grad_out, &nodes, &masks, node.domain_id);
    }
}
```

## 7.5 稀疏 optimizer —— 跳过参数更新

```rust
// oyta-optim/src/sparse_adam.rs
pub fn step_sparse(&mut self, active_ids: &[u32]) {
    for (domain_id, params) in self.param_pool.iter_mut() {
        if !active_ids.contains(domain_id) { continue; }
        for param in params {
            if param.grad().is_none() { continue; }
            // 执行标准 AdamW / SGD 更新
            ...
        }
    }
}
```

## 7.6 稀疏掩码三层结构（0.6.8 完整设计）

**MaskLevel 枚举：**
```rust
// oyta-router/src/sparse_mask.rs
pub enum MaskLevel {
    DomainMask,                                    // Level 1: 领域级（整个专家激活/跳过）
    ChannelMask { expert_domain_id: u32 },         // Level 2: 通道级（领域内特征选择/Early Exit）
    SequenceMask { expert_domain_id: u32, channel: usize }, // Level 3: 序列级（变长/padding）
}

pub struct SparseMaskSet {
    pub domain_mask: Vec<bool>,                    // len=num_domains, true=激活
    pub channel_masks: HashMap<u32, Vec<f32>>,    // domain_id → mask[f32; num_channels] (0=参与,1=mask)
    pub sequence_masks: HashMap<(u32, usize), Vec<f32>>, // (domain_id, channel) → mask[f32; seq_len]
}
```

**RoutingStrategy 生成 Level 1 mask（DomainRouter 主入口）：**
```rust
pub fn generate_domain_mask(&self, domain_weights, total_domains) -> Vec<bool> {
    match self {
        TopK { k } => {
            // 降序排序 → 前 K 个标记激活 → 保证至少激活一个（空集 fallback）
        }
        Threshold { threshold } => {
            // 权重 > 阈值 → 激活 → 保证至少激活一个
        }
    }
}

pub fn compute_normalized_weights(&self, domain_weights) -> (Vec<u32>, Vec<f32>) {
    // 取激活领域 → softmax 归一化 → 返回 (active_ids, normalized_weights)
    // normalized_weights 供 FeatureFusion gating 用
}
```

**Op 枚举掩码变体（稀疏 mask 传递到 autograd）：**
```rust
// oyta-autograd/src/op.rs
pub enum Op {
    // 原有变体保持不变
    Conv1d { kernel_size, stride, padding },
    Linear, BatchNorm1d, MaxPool1d, Matmul, Relu, Gelu, ...

    // ========== Phase 0 新增：带稀疏掩码的算子变体 ==========
    Conv1dMasked { channel_mask: Vec<f32> },
    BatchNorm1dMasked { channel_mask: Vec<f32> },
    Conv1dWithSequenceMask { sequence_mask: Vec<f32> },
    Router { domain_mask: Vec<bool>, normalized_weights: Vec<f32> },
}
```

**算子内部 mask 应用（backward 时乘 (1-mask)）：**
```rust
// Op::Conv1dMasked backward 分支
Op::Conv1dMasked { channel_mask } => {
    // grad_out_masked[f32, out_c, out_len] *= (1 - channel_mask[f])
    // 然后调标准 Conv1d backward
}
Op::BatchNorm1dMasked { channel_mask } => {
    // grad_x 的 feature f 如果被 mask → 全 batch 该特征清零
    // grad_gamma 的 feature f 如果被 mask → 清零
}
```

## 7.7 完整稀疏数据流图（CNNExpert 内部示例）

```
激活专家 (domain_id=2):
┌─────────────────────────────────────────────────┐
│ x [batch, seq_len, features]                    │
│   │                                             │
│   ▼ Conv1d (kernel=3, stride=1, padding=1)      │ ← 底层 Conv1dOps 算子
│   │                                             │
│   ▼ BatchNorm1d (训练态, 缓存 x̂ + inv_std)      │ ← 底层 BatchNormOps
│   │                                             │
│   ▼ ReLU                                        │
│   │                                             │
│   ▼ MaxPool1d (kernel=2, stride=2)              │ ← 底层 PoolOps, 缓存 argmax
│   │                                             │
│   ▼ Linear                                      │
│   │                                             │
│   ▼ expert_output                               │
└─────────────────────────────────────────────────┘

未激活专家 (domain_id=3):
┌─────────────────────────────────────────────────┐
│ 全部跳过：forward 不执行 → backward 不执行       │
│ → 中间张量不分配 → 梯度 = None → optimizer 跳过  │
└─────────────────────────────────────────────────┘
```

---

# 八、四层分层目录规范（所有 crate 统一遵守）

> 架构设计文档要求"模块化代码"，结合实际开发中总结的最佳实践，所有 crate 统一四层分层

```
oyta-{crate}/src/
├── lib.rs                   ← 顶层统一 pub mod 声明 + pub use + prelude
│                            ↑ 只写模块声明和导出，零实现代码
│
├── types/                   ← 数据类型层（纯结构体定义，无业务逻辑）
│   ├── mod.rs              ← 只 pub mod xxx;（零 pub use，零实现）
│   └── xxx.rs              ← Domain / RawInput / ContentAnalysis / Parameter ...
│
├── traits/                  ← 接口层（只 trait 定义，零实现代码）
│   ├── mod.rs              ← 只 pub mod xxx;（零 pub use，零实现）
│   └── xxx.rs              ← ContentAnalyzer / DomainExpert / Optimizer / LossFn ...
│
├── utils/                   ← 工具层（底层可复用组件，被 impl 层依赖）
│   ├── mod.rs              ← 只 pub mod xxx;（零 pub use，零实现）
│   └── xxx.rs              ← FeatureEncoder / MiniMLP / SyntaxPattern / MetaRouter ...
│
└── analyzers/ (或 impl/)    ← 实现层（trait 的具体实现）
    ├── mod.rs              ← 只 pub mod xxx;（零 pub use，零实现）
    └── xxx.rs              ← RuleBasedAnalyzer / HybridAnalyzer / ExpertExecutor ...
```

### 每层职责边界（严格单向依赖）

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ analyzers/ |
| traits/ | types/ | utils/ analyzers/ |
| utils/ | types/ | traits/ analyzers/ |
| analyzers/ | types/ + traits/ + utils/ | 反向依赖自己 |

### mod.rs 铁律（5 条禁止事项第 5 条细化）

**所有 mod.rs（包括 types/mod.rs, utils/mod.rs, analyzers/mod.rs）只允许写：**
```rust
//! xxx 层说明

pub mod xxx1;
pub mod xxx2;
```

**禁止在 mod.rs 里写：**
1. ❌ `pub use xxx::Xxx;`（导出统一走 lib.rs 从子文件直接 pub use）
2. ❌ 任何函数/结构体/trait 实现
3. ❌ `use` 语句（除了注释里说明）

### lib.rs 正确的导出方式

```rust
// lib.rs —— 从子文件直接 pub use，不经过 mod.rs
pub mod types;
pub mod traits;
pub mod utils;
pub mod analyzers;

pub use types::domain::Domain;
pub use types::raw_input::RawInput;
pub use traits::analyzer::ContentAnalyzer;
pub use utils::mini_mlp::MiniMLP;
pub use analyzers::hybrid::HybridAnalyzer;

pub mod prelude {
    pub use crate::*;
    pub use oyta_tensor::prelude::*;
}
```
