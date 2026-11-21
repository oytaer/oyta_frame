# oyta-expert 开发文档（Phase 3）

> crate：`oyta-expert/` | 依赖：**`oyta-tensor = "0.1.1"` + `oyta-router`** | 职责：DomainExpert trait + ExpertExecutor 框架层稀疏控制 + 4 种专家

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-expert"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
oyta-router = { version = "0.1.0", path = "../oyta-router" }
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs

```rust
pub mod domain_expert;
pub mod expert_executor;
pub mod linear;
pub mod norm;
pub mod dropout;
pub mod experts;

pub use domain_expert::DomainExpert;
pub use expert_executor::ExpertExecutor;
pub use linear::Linear;
pub use norm::{LayerNorm, RMSNorm};
pub use dropout::set_training_mode;
pub use experts::{MLPExpert, CNN1DExpert, RNNExpert, TransformerExpert};

pub mod prelude {
    pub use crate::{
        DomainExpert, ExpertExecutor, Linear, LayerNorm, RMSNorm,
        set_training_mode,
        MLPExpert, CNN1DExpert, RNNExpert, TransformerExpert,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool, RoutingResult};
}
```

---

# 三、src/linear.rs（全部调用 oyta_tensor API，不手写 matmul）

```rust
//! Linear 层：y = x @ W^T + b
//!
//! 全部算子走 oyta_tensor API：
//!   w.transpose(0, 1).matmul(x) → wx
//!   wx.add(&b.broadcast_to(wx.shape())) → y
//!
//! 参数初始化：用 Tensor::<f32, Cpu>::randn（oyta-tensor 工厂方法）
//!   Kaiming 均匀分布：先 randn 再乘 scale = sqrt(6/fan_in)

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};

pub struct Linear {
    weight: Parameter,  // [out_features, in_features]
    bias:   Parameter,  // [out_features]
}

impl Linear {
    /// 构造 Linear（参数自动初始化）
    pub fn new(domain_id: u32, module: &str, in_features: usize, out_features: usize) -> Self {
        // Kaiming 均匀分布初始化（调用 oyta_tensor 工厂方法 randn）
        let scale = (6.0 / in_features as f32).sqrt();
        let w_data: Vec<f32> = Tensor::<f32, Cpu>::randn([out_features, in_features])
            .data_slice().iter().map(|v| v * scale).collect();
        let w_tensor = Tensor::<f32, Cpu>::from_data(w_data, [out_features, in_features]).unwrap();
        let b_tensor = Tensor::<f32, Cpu>::zeros([out_features]);

        Self {
            weight: build_param(domain_id, module, "weight", w_tensor),
            bias:   build_param(domain_id, module, "bias",   b_tensor),
        }
    }

    /// 前向：全调 oyta_tensor API
    /// y = x @ W^T + b
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let w = self.weight.data.lock().unwrap().clone();   // [out, in]
        let b = self.bias.data.lock().unwrap().clone();     // [out]

        // (a) W^T：oyta_tensor 有没有 transpose？
        // oyta-core/src/tensor/ops.rs 里没看到 Tensor::transpose 方法
        // oyta-device::MatmulOps 有 transa/transb 参数（Backend trait）
        // 所以更高效的方式是直接用 w 和 x 做 matmul，指定 W 需要转置
        // 但 Tensor::matmul API 不暴露 transa/transb
        // 备选方案：自己写转置函数（纯 Vec 操作，不是算子）
        let wt = Self::transpose(&w);  // [in, out]

        // (b) wx = x @ W^T：[batch, in] @ [in, out] → [batch, out]
        let wx = wt.matmul(x).expect("Linear::forward matmul 失败");

        // (c) 偏置广播加到每一行：[batch, out] += [out]
        // oyta_tensor::Tensor 有没有 broadcast_add？
        // 看 ops.rs：有 add 方法，但要求 shape 相同
        // 所以先手动构造完整 shape 的 bias（纯 Vec 操作）
        let batch = wx.shape().dims[0];
        let out_feat = b.shape().dims[0];
        let mut b_broadcast = Vec::with_capacity(batch * out_feat);
        let bd = b.data_slice();
        for _n in 0..batch { b_broadcast.extend_from_slice(bd); }
        let b_full = Tensor::<f32, Cpu>::from_data(b_broadcast, [batch, out_feat]).unwrap();

        wx.add(&b_full).expect("Linear::forward bias add 失败")
    }

    /// 纯 Vec 操作转置（不碰 oyta_tensor，只是数据搬运）
    fn transpose(t: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let [rows, cols] = t.shape().dims[..] else { panic!("transpose 需要 2D") };
        let d = t.data_slice();
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows { for j in 0..cols { out[j * rows + i] = d[i * cols + j]; } }
        Tensor::<f32, Cpu>::from_data(out, [cols, rows]).unwrap()
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
```

---

# 四、src/norm.rs（直接调 oyta_tensor::Tensor::layernorm）

```rust
//! LayerNorm / RMSNorm（直接调 oyta_tensor API）
//!
//! Tensor::<f32, Cpu>::layernorm(gamma, beta, eps) 已经在 ops.rs 实现
//! 这里只做 Parameter 包装 + 初始化

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};

pub struct LayerNorm {
    num_features: usize,
    eps: f32,
    gamma: Parameter,  // 初始化为 1
    beta:  Parameter,  // 初始化为 0
}

impl LayerNorm {
    pub fn new(domain_id: u32, module: &str, num_features: usize) -> Self {
        let g = Tensor::<f32, Cpu>::from_data(vec![1.0f32; num_features], [num_features]).unwrap();
        let b = Tensor::<f32, Cpu>::zeros([num_features]);
        Self {
            num_features, eps: 1e-5,
            gamma: build_param(domain_id, module, "gamma", g),
            beta:  build_param(domain_id, module, "beta", b),
        }
    }

    /// 直接调 oyta_tensor::Tensor::layernorm API
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let g = self.gamma.data.lock().unwrap().clone();
        let b = self.beta.data.lock().unwrap().clone();
        x.layernorm(&g, &b, self.eps).expect("LayerNorm::forward 失败")
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

/// RMSNorm（只缩放，无中心化，更快）
pub struct RMSNorm {
    num_features: usize,
    eps: f32,
    gamma: Parameter,
}

impl RMSNorm {
    pub fn new(domain_id: u32, module: &str, num_features: usize) -> Self {
        let g = Tensor::<f32, Cpu>::from_data(vec![1.0f32; num_features], [num_features]).unwrap();
        Self { num_features, eps: 1e-6, gamma: build_param(domain_id, module, "gamma", g) }
    }

    /// oyta_tensor 有没有 rmsnorm？看 oyta_ops/src/norm/layernorm.rs 有 rmsnorm
    /// 或者用 Tensor 方法... ops.rs 里没看到 rmsnorm 方法
    /// 备选：自己算（逐样本 rms → 缩放，纯 Vec 操作，不是算子手写）
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let rank = x.rank();
        let inner = x.shape().dims[rank - 1];
        let outer = x.numel() / inner;
        let d = x.contiguous();
        let dd = d.data_slice();
        let g = self.gamma.data.lock().unwrap().clone();
        let gd = g.data_slice();

        let mut out = vec![0.0f32; x.numel()];
        for row in 0..outer {
            let row_start = row * inner;
            // 计算 RMS
            let mut sum_sq = 0.0f64;
            for f in 0..inner { sum_sq += (dd[row_start + f] as f64).powi(2); }
            let rms = (sum_sq / inner as f64 + self.eps as f64).sqrt();
            let inv_rms = 1.0 / rms as f32;
            // 缩放 × gamma
            for f in 0..inner {
                out[row_start + f] = dd[row_start + f] * inv_rms * gd[f];
            }
        }
        Tensor::<f32, Cpu>::from_data(out, x.shape().clone()).unwrap()
    }

    pub fn parameters(&self) -> Vec<Parameter> { vec![self.gamma.clone()] }
}
```

---

# 五、src/dropout.rs（调用 oyta_ops 的 set_training_mode）

```rust
//! Dropout 训练态切换（直接 re-export oyta_ops）
//! oyta_ops::dropout 提供 set_training_mode / is_training_mode
//! 训练层内部用 is_training_mode() 判断是否启用 Dropout

pub use oyta_tensor::set_training_mode;
pub use oyta_tensor::is_training_mode;

/// 简化版 Dropout（只做判断 + 掩码）
pub fn apply_dropout(x: &oyta_tensor::Tensor<f32, oyta_tensor::Cpu>, rate: f32)
    -> oyta_tensor::Tensor<f32, oyta_tensor::Cpu> {
    if !is_training_mode() || rate <= 0.0 { return x.clone(); }
    // Phase 1 先不真正做 Dropout（需要 rand crate），直接返回
    // Phase 1.5 补随机掩码 + scale 1/(1-rate)
    let _ = rate;
    x.clone()
}
```

---

# 六、src/domain_expert.rs

```rust
//! DomainExpert trait：所有专家基类
use oyta_tensor::prelude::*;
use oyta_router::{Parameter, SparseParameterPool};

pub trait DomainExpert: Send + Sync {
    fn domain_id(&self) -> u32;
    fn forward(&self, input: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;
    fn parameters(&self) -> Vec<Parameter>;
    fn is_training(&self) -> bool;
    fn train(&mut self) { set_training_mode(true); }
    fn eval(&mut self)  { set_training_mode(false); }
    fn output_dim(&self) -> usize;
    fn input_dim(&self) -> usize;
    fn register(&self, pool: &mut SparseParameterPool) {
        pool.register_domain(self.domain_id(), self.parameters());
    }
}
```

---

# 七、src/expert_executor.rs（框架层稀疏控制）

```rust
//! ExpertExecutor：框架层稀疏控制流（核心！）
//!
//! 稀疏发生在 for 循环里 continue 跳过未激活专家
//! 未激活专家 forward 零调用、不分配中间 Tensor

use std::collections::HashMap;
use oyta_tensor::prelude::*;
use oyta_router::{RoutingResult, SparseParameterPool, Parameter};
use crate::domain_expert::DomainExpert;

pub struct ExpertExecutor {
    pub experts: Vec<Box<dyn DomainExpert>>,
    pub param_pool: SparseParameterPool,
}

impl ExpertExecutor {
    pub fn new(experts: Vec<Box<dyn DomainExpert>>) -> Self {
        let mut pool = SparseParameterPool::new();
        for e in &experts { pool.register_domain(e.domain_id(), e.parameters()); }
        Self { experts, param_pool: pool }
    }

    /// 稀疏 forward：只执行激活专家
    pub fn forward_sparse(
        &self,
        input: &Tensor<f32, Cpu>,
        result: &RoutingResult,
    ) -> Vec<(u32, Tensor<f32, Cpu>)> {
        let mut outputs: Vec<(u32, Tensor<f32, Cpu>)> = Vec::with_capacity(result.active_indices.len());
        for expert in &self.experts {
            let did = expert.domain_id();
            if !result.active_indices.contains(&did) {
                continue;  // ===== 框架层核心稀疏点：未激活专家零执行 =====
            }
            outputs.push((did, expert.forward(input)));
        }
        outputs
    }

    pub fn train_all(&mut self) { for e in self.experts.iter_mut() { e.as_mut().train(); } }
    pub fn eval_all(&mut self)  { for e in self.experts.iter_mut() { e.as_mut().eval(); } }
}
```

---

# 八、src/experts/mod.rs + src/experts/mlp.rs

```rust
// src/experts/mod.rs
pub mod mlp;
pub mod cnn1d;
pub mod transformer;
pub mod rnn;
pub use mlp::MLPExpert;
pub use cnn1d::CNN1DExpert;
pub use transformer::TransformerExpert;
pub use rnn::RNNExpert;

// src/experts/mlp.rs
//! MLPExpert：Linear → LayerNorm → GELU → Linear
//!
//! 全部底层算子调用 oyta_tensor API
//!   Linear::forward（内部 matmul + add）
//!   LayerNorm::forward（内部 Tensor::layernorm）
//!   Tensor::gelu()（直接调 oyta_tensor）

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::domain_expert::DomainExpert;
use crate::linear::Linear;
use crate::norm::LayerNorm;

pub struct MLPExpert {
    did: u32,
    in_dim: usize,
    out_dim: usize,
    hidden: Linear,     // in → hidden
    ln: LayerNorm,
    out: Linear,        // hidden → out
}

impl MLPExpert {
    pub fn new(did: u32, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            did, in_dim, out_dim: out_dim,
            hidden: Linear::new(did, "mlp.hidden", in_dim, hidden_dim),
            ln: LayerNorm::new(did, "mlp.ln", hidden_dim),
            out: Linear::new(did, "mlp.out", hidden_dim, out_dim),
        }
    }
}

impl DomainExpert for MLPExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_dim }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { is_training_mode() }

    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // 全调 oyta_tensor API（Linear 内部用 matmul，LayerNorm 内部用 layernorm，gelu 直接调）
        let h1 = self.hidden.forward(x);
        let h2 = self.ln.forward(&h1);
        let h3 = h2.gelu().expect("MLP::forward gelu 失败");
        self.out.forward(&h3)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.hidden.parameters());
        all.extend(self.ln.parameters());
        all.extend(self.out.parameters());
        all
    }
}
```

---

# 九、src/experts/transformer.rs（骨架，核心算子全调 oyta_tensor）

```rust
//! TransformerExpert：Pre-LN Transformer（2 层）
//!
//! 核心算子：FlashAttention（oyta_ops::attention::flash::flash_attention_multihead）
//!         + LayerNorm（Tensor::layernorm）+ Linear

use oyta_tensor::prelude::*;
use oyta_tensor::flash_attn::flash_attention_multihead;
use oyta_router::Parameter;
use crate::domain_expert::DomainExpert;
use crate::linear::Linear;
use crate::norm::LayerNorm;

pub struct TransformerExpert {
    did: u32,
    in_dim: usize,
    out_dim: usize,
    num_heads: usize,
    // 每层 Pre-LN Transformer：LN → QKV → Attn → LN → FFN
    // 骨架先放 1 层简化版，核心算子展示正确调用方式
    ln1: LayerNorm,
    qkv: Linear,    // d_model → 3*d_model
    o_proj: Linear, // d_model → d_model
    ln2: LayerNorm,
    ffn1: Linear,   // d_model → 4*d_model
    ffn2: Linear,   // 4*d_model → d_model
    proj: Linear,   // d_model → out_dim
}

impl TransformerExpert {
    pub fn new(did: u32, in_dim: usize, d_model: usize, out_dim: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        Self {
            did, in_dim, out_dim, num_heads,
            ln1: LayerNorm::new(did, "trans.ln1", d_model),
            qkv: Linear::new(did, "trans.qkv", d_model, 3*d_model),
            o_proj: Linear::new(did, "trans.o", d_model, d_model),
            ln2: LayerNorm::new(did, "trans.ln2", d_model),
            ffn1: Linear::new(did, "trans.ffn1", d_model, 4*d_model),
            ffn2: Linear::new(did, "trans.ffn2", 4*d_model, d_model),
            proj: Linear::new(did, "trans.proj", d_model, out_dim),
        }
    }
}

impl DomainExpert for TransformerExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_dim }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { is_training_mode() }

    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // Phase 1 骨架：先跑 LN → FFN → proj
        // FlashAttention 调用（展示正确 API）：
        // let qkv = self.qkv.forward(&ln1_out);
        // let attn_out = flash_attention_multihead(&q, &k, &v, self.num_heads, false, 1.0/(d_k.sqrt() as f32));
        // let out = self.o_proj.forward(&attn_out);
        // 然后 LN → FFN → proj
        let ln1 = self.ln1.forward(x);
        let ffn1_out = self.ffn1.forward(&ln1).gelu().unwrap();
        let ffn2_out = self.ffn2.forward(&ffn1_out);
        self.proj.forward(&ffn2_out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.ln1.parameters());
        all.extend(self.qkv.parameters());
        all.extend(self.o_proj.parameters());
        all.extend(self.ln2.parameters());
        all.extend(self.ffn1.parameters());
        all.extend(self.ffn2.parameters());
        all.extend(self.proj.parameters());
        all
    }
}
```

---

# 十、oyta-expert 验收标准

```
[ ] cargo check --package oyta-expert 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1"（无 rust-version、无分散底层依赖）
[ ] 所有算子调用 oyta_tensor API：
    Linear::forward 用 Tensor::matmul ✓
    LayerNorm::forward 用 Tensor::layernorm ✓
    MLPExpert.forward 用 Tensor::gelu ✓
    TransformerExpert 引用 flash_attention_multihead（骨架）✓
[ ] ExpertExecutor.forward_sparse 里 continue 跳过未激活专家
[ ] 每个函数有中文注释
```

---

# 十一、Phase 0 Conv1d/BatchNorm1d/MaxPool1d Backend trait 完整定义（架构 §0）

> 架构设计文档 §0 新增三种 Phase 0 算子，全部通过 oyta-tensor Backend trait 扩展实现。框架层（oyta-expert）只调高层接口，不手搓算子内部循环。以下 trait 签名可直接指导 oyta-device 实现。

## 11.1 Conv1dOps 完整方法签名

```rust
//! oyta-device/src/backend/conv1d.rs
//! Conv1d 一维卷积 Backend trait（标准 NCHW 布局）

use oyta_core::backend::BackendBase;

pub trait Conv1dOps<T: Float>: BackendBase {
    /// 前向传播：标准 NCHW 一维卷积
    ///
    /// 数学公式：out[n, oc, o] = bias[oc] + Σ_{ic, k} weight[oc, ic, k] × x[n, ic, o*stride + k - padding]
    /// 其中 out_len = (seq_len + 2*padding - k_size) / stride + 1
    ///
    /// # 参数
    /// - x: 输入 [batch, in_c, seq_len]
    /// - w: 权重 [out_c, in_c, k_size]
    /// - b: 偏置 [out_c]
    /// - out: 输出 [batch, out_c, out_len]（预分配）
    fn conv1d(
        x: *const T, w: *const T, b: *const T, out: *mut T,
        batch: usize, in_c: usize, seq_len: usize,
        out_c: usize, k_size: usize, stride: usize, padding: usize,
    );

    /// 反向传播：同时计算 ∂L/∂x, ∂L/∂w, ∂L/∂b
    ///
    /// # 参数
    /// - x: 原始输入（forward 时缓存）
    /// - w: 原始权重
    /// - go: 输出梯度 [batch, out_c, out_len]
    /// - gx: 输入梯度 [batch, in_c, seq_len]
    /// - gw: 权重梯度 [out_c, in_c, k_size]
    /// - gb: 偏置梯度 [out_c]
    fn conv1d_backward(
        x: *const T, w: *const T, go: *const T,
        gx: *mut T, gw: *mut T, gb: *mut T,
        batch: usize, in_c: usize, seq_len: usize,
        out_c: usize, k_size: usize, stride: usize, padding: usize,
    );
}
```

**backward 推导要点**：
- `∂L/∂w[oc, ic, k] = Σ_{n, o} go[n, oc, o] × x[n, ic, o*stride + k - padding]`（互相关）
- `∂L/∂x[n, ic, t] = Σ_{oc, o, k} go[n, oc, o] × w[oc, ic, k]`（dilated 反卷积）
- 中间量必须用 f64 累加器防 batch 大时精度损失
- rayon 并行按 `batch × out_channels` 分块

## 11.2 BatchNorm1dOps 完整方法签名

```rust
//! oyta-device/src/backend/batchnorm1d.rs
//! BatchNorm1d 批归一化 Backend trait

use oyta_core::backend::BackendBase;

pub trait BatchNorm1dOps<T: Float>: BackendBase {
    /// 训练态前向（同时缓存 BN backward 所需数据 + 更新 running 统计）
    ///
    /// # 数学
    /// μ = (1/N) Σ x_i,  σ² = (1/N) Σ (x_i - μ)²
    /// x̂ = (x - μ) / √(σ² + ε),  ŷ = γ ⊙ x̂ + β
    /// running_mean = (1-m)·running_mean + m·μ  （指数移动平均）
    /// running_var  = (1-m)·running_var  + m·σ²
    ///
    /// # 缓存要求（forward 时必须写入）
    /// - cached_xhat   [batch, num_features] — 归一化中间结果 x̂
    /// - cached_inv_std [num_features]       — 每个特征的 inv_std = 1/√(σ²+ε)
    ///   ⚠️ 没有这两个缓存，backward 无法高效计算（O(N²) Jacobian 不可接受）
    ///
    /// # Running 统计稀疏训练注意点
    /// - 框架层只对激活专家调此方法 → 未激活专家的 BN running 统计自动冻结
    /// - 初始化 running_mean=0, running_var=1, gamma=1, beta=0 → BN 输出恒等变换
    fn batchnorm1d_train(
        x: *const T, gamma: *const T, beta: *const T,
        running_mean: *mut T, running_var: *mut T,
        out: *mut T, cached_xhat: *mut T, cached_inv_std: *mut T,
        batch: usize, num_features: usize,
        eps: T, momentum: T,
    );

    /// 推理态前向（只读 running_mean / running_var，无缓存写入）
    fn batchnorm1d_eval(
        x: *const T, gamma: *const T, beta: *const T,
        running_mean: *const T, running_var: *const T,
        out: *mut T,
        batch: usize, num_features: usize,
        eps: T,
    );

    /// 反向传播（训练态简化公式，避免 O(N²) Jacobian）
    ///
    /// # 完整简化公式
    ///   scaled_go[n] = go[n] · γ_f
    ///   mean_sgo = (1/N) Σ scaled_go[n]
    ///   mean_sx  = (1/N) Σ scaled_go[n] · x̂[n]
    ///   dL/dx[m,f] = inv_std[f] × [ scaled_go[m] - mean_sgo - x̂[m] × mean_sx ]
    ///   dL/dβ[f]  = Σ go[n, f]
    ///   dL/dγ[f]  = Σ go[n, f] · x̂[n, f]
    ///
    /// # 参数
    /// - xhat: forward 缓存的 x̂ [batch, num_features]
    /// - inv_std: forward 缓存的 inv_std [num_features]
    fn batchnorm1d_backward(
        go: *const T, xhat: *const T, inv_std: *const T, gamma: *const T,
        gx: *mut T, ggamma: *mut T, gbeta: *mut T,
        batch: usize, num_features: usize,
    );
}
```

## 11.3 PoolOps 完整方法签名

```rust
//! oyta-device/src/backend/pool.rs
//! MaxPool1d / AvgPool1d Backend trait

use oyta_core::backend::BackendBase;

pub trait PoolOps<T: Float>: BackendBase {
    /// MaxPool1d 前向（必须同时写入 argmax 索引数组，backward 必需）
    ///
    /// 稀疏性对比：MaxPool backward 天然稀疏（只有 argmax 位置有梯度），AvgPool 稠密
    /// # 参数
    /// - x: [batch, channels, seq_len]
    /// - out: [batch, channels, out_len]（预分配）
    /// - argmax: [batch, channels, out_len]（i32，存储输入位置索引）
    fn maxpool1d(
        x: *const T, out: *mut T, argmax: *mut i32,
        batch: usize, channels: usize, seq_len: usize,
        k_size: usize, stride: usize, padding: usize,
    );

    /// MaxPool1d 反向（只对 argmax 位置写入 grad_x，其余为零）
    /// # 参数
    /// - grad_out: [batch, channels, out_len]
    /// - argmax: forward 缓存的 argmax
    /// - grad_x: [batch, channels, seq_len]
    fn maxpool1d_backward(
        grad_out: *const T, argmax: *const i32, grad_x: *mut T,
        batch: usize, channels: usize, seq_len: usize,
        k_size: usize, stride: usize, padding: usize,
    );

    /// AvgPool1d 前向（无 argmax 缓存需求）
    fn avgpool1d(
        x: *const T, out: *mut T,
        batch: usize, channels: usize, seq_len: usize,
        k_size: usize, stride: usize, padding: usize,
    );

    /// AvgPool1d 反向（稠密：每个输入位置平分梯度）
    fn avgpool1d_backward(
        grad_out: *const T, grad_x: *mut T,
        batch: usize, channels: usize, seq_len: usize,
        k_size: usize, stride: usize, padding: usize,
    );
}
```

---

# 十二、四层分层目录规范（对齐 00-overview §八）

oyta-expert 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层，严格遵守 00-overview §八 的职责边界和 mod.rs 铁律。

## 12.1 目标目录结构

```
oyta-expert/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude（只声明+导出）
│
├── types/                       ← 数据类型层（纯结构体，零业务逻辑）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   └── expert_cache.rs         ← BN 缓存结构 { cached_xhat, cached_inv_std }、Pool argmax 缓存结构
│
├── traits/                      ← 接口层（只 trait 定义，零实现）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   └── domain_expert.rs        ← DomainExpert trait（domain_id / forward / parameters / train / eval / output_dim / input_dim / register）
│
├── utils/                       ← 工具层（被 impl 层依赖的底层可复用组件）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   ├── linear.rs               ← Linear 模块（参数包装 + 初始化 + forward 调 oyta_tensor matmul + add）
│   ├── norm.rs                 ← LayerNorm / RMSNorm（Tensor::layernorm）
│   ├── batchnorm1d.rs          ← BatchNorm1d（Phase 0 新增：BN 训练态/推理态 + 缓存）
│   ├── dropout.rs              ← re-export oyta_tensor set_training_mode + apply_dropout
│   └── conv1d.rs               ← Conv1d（Phase 0 新增：Forward + 缓存输入 x 供 backward）
│
└── impl/                        ← 实现层（trait 的具体实现）
    ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
    ├── expert_executor.rs      ← ExpertExecutor（框架层稀疏控制 for 循环 continue）
    ├── mlp_expert.rs           ← MLPExpert（Linear → LayerNorm → GELU → Linear）
    ├── cnn1d_expert.rs         ← CNN1DExpert（Conv1d → BN1d → ReLU → MaxPool1d → Linear）
    ├── rnn_expert.rs           ← RNNExpert（骨架）
    └── transformer_expert.rs   ← TransformerExpert（Pre-LN + FlashAttention 骨架）
```

## 12.2 每层职责边界（严格单向依赖）

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + traits/ + utils/ | 反向依赖自己 |

## 12.3 mod.rs 铁律（对齐 00-overview §八）

**所有 mod.rs 只允许写：**
```rust
//! types 层说明

pub mod expert_cache;
```

**禁止在 mod.rs 里写**：
1. ❌ `pub use xxx::Xxx;`（导出统一走 lib.rs 从子文件直接 pub use）
2. ❌ 任何函数/结构体/trait 实现
3. ❌ `use` 语句

## 12.4 lib.rs 正确的导出方式

```rust
//! oyta-expert：Phase 3 框架层 crate
//! 四层分层：types → traits → utils → impl

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// 从子文件直接 pub use，不经过中间 mod.rs
pub use types::expert_cache::{BatchNormCache, Conv1dCache, PoolCache};
pub use traits::domain_expert::DomainExpert;
pub use utils::linear::Linear;
pub use utils::norm::{LayerNorm, RMSNorm};
pub use utils::batchnorm1d::BatchNorm1d;
pub use utils::dropout::{set_training_mode, is_training_mode, apply_dropout};
pub use utils::conv1d::Conv1d;
pub use impls::expert_executor::ExpertExecutor;
pub use impls::mlp_expert::MLPExpert;
pub use impls::cnn1d_expert::CNN1DExpert;
pub use impls::rnn_expert::RNNExpert;
pub use impls::transformer_expert::TransformerExpert;

pub mod prelude {
    pub use crate::*;
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool, RoutingResult};
}
```

---

# 十三、验收标准（扩充版：含四层分层 + Backend trait + 算子缓存要求）

```
## 基础验收（原有）
[ ] cargo check --package oyta-expert 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1" + oyta-router（无 rust-version、无分散底层依赖）
[ ] 所有算子调用 oyta_tensor API：
    Linear::forward → Tensor::matmul + Tensor::add ✓
    LayerNorm::forward → Tensor::layernorm ✓
    MLPExpert.forward → Tensor::gelu ✓
    TransformerExpert 引用 flash_attention_multihead（骨架）✓
[ ] ExpertExecutor.forward_sparse 里 continue 跳过未激活专家
[ ] 每个函数有中文注释

## 追加：Backed trait 验收
[ ] Conv1dOps trait 签名完整：conv1d(x, w, b, out, batch, in_c, seq_len, out_c, k_size, stride, padding) + conv1d_backward(x, w, go, gx, gw, gb, 同参数)
[ ] BatchNorm1dOps trait 签名完整：batchnorm1d_train(..., cached_xhat, cached_inv_std, ...) + batchnorm1d_eval(...) + batchnorm1d_backward(go, xhat, inv_std, gamma, gx, ggamma, gbeta, ...)
[ ] PoolOps trait 签名完整：maxpool1d(x, out, argmax, ...) + maxpool1d_backward(grad_out, argmax, grad_x, ...) + avgpool1d + avgpool1d_backward
[ ] BatchNorm 缓存要求明确：forward 必须写入 cached_xhat [batch, num_features] + cached_inv_std [num_features]
[ ] BN running 统计稀疏训练注意点：未激活专家 BN 不更新 running_mean/running_var（框架层只调激活专家的 train）
[ ] MaxPool argmax 缓存要求明确：forward 必须写入 argmax [batch, channels, out_len]（i32 位置索引）

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use（types::expert_cache::BatchNormCache 等），不经过中间 mod.rs
[ ] 依赖方向单向：types → traits/utils → impls（无反向依赖）
[ ] Linear / LayerNorm / RMSNorm / BatchNorm1d / Conv1d / Dropout 归入 utils/ 层
[ ] DomainExpert trait 归入 traits/ 层
[ ] ExpertExecutor + MLPExpert + CNN1DExpert + RNNExpert + TransformerExpert 归入 impl/ 层
[ ] expert_cache.rs（BN 缓存结构 + Pool argmax 缓存结构）归入 types/ 层
```
