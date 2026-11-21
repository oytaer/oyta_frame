# oyta-fusion 开发文档（Phase 4）

> crate：`oyta-fusion/` | 依赖：**`oyta-tensor = "0.1.1"` + `oyta-expert`** | 职责：三种融合策略 + TaskHead

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-fusion"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
oyta-expert = { version = "0.1.0", path = "../oyta-expert" }
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs

```rust
pub mod fusion;
pub mod weighted_sum;
pub mod concat_linear;
pub mod cross_attention;
pub mod task_head;

pub use fusion::{FusionStrategy, FeatureFusion};
pub use weighted_sum::WeightedSumFusion;
pub use concat_linear::ConcatLinearFusion;
pub use cross_attention::CrossAttentionFusion;
pub use task_head::{TaskHeadBase, ClassificationHead, RegressionHead};

pub mod prelude {
    pub use crate::{
        FusionStrategy, FeatureFusion,
        WeightedSumFusion, ConcatLinearFusion, CrossAttentionFusion,
        TaskHeadBase, ClassificationHead, RegressionHead,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::Parameter;
    pub use oyta_expert::Linear;
}
```

---

# 三、src/fusion.rs

```rust
//! FusionStrategy 枚举 + FeatureFusion 主类
use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::weighted_sum::WeightedSumFusion;
use crate::concat_linear::ConcatLinearFusion;
use crate::cross_attention::CrossAttentionFusion;

#[derive(Debug, Clone)]
pub enum FusionStrategy {
    WeightedSum,
    ConcatLinear { proj_dim: usize },
    CrossAttention { num_heads: usize, dropout: f32, proj_dim: usize },
}

pub struct FeatureFusion {
    strategy: FusionStrategy,
    fused_dim: usize,
    weighted_sum: Option<WeightedSumFusion>,
    concat_linear: Option<ConcatLinearFusion>,
    cross_attention: Option<CrossAttentionFusion>,
    did: u32,
    training: bool,
}

impl FeatureFusion {
    pub fn new(strategy: FusionStrategy, did: u32, expert_dim: usize) -> Self {
        let fused_dim = match &strategy {
            FusionStrategy::WeightedSum => expert_dim,
            FusionStrategy::ConcatLinear { proj_dim } => *proj_dim,
            FusionStrategy::CrossAttention { proj_dim, .. } => *proj_dim,
        };
        Self {
            fused_dim, strategy: strategy.clone(), did,
            weighted_sum: matches!(strategy, FusionStrategy::WeightedSum).then(|| WeightedSumFusion::new(expert_dim)),
            concat_linear: if let FusionStrategy::ConcatLinear { proj_dim } = &strategy {
                Some(ConcatLinearFusion::new(did, expert_dim, *proj_dim))
            } else { None },
            cross_attention: if let FusionStrategy::CrossAttention { num_heads, dropout, proj_dim } = &strategy {
                Some(CrossAttentionFusion::new(did, expert_dim, *num_heads, *dropout, *proj_dim))
            } else { None },
            did, training: true,
        }
    }

    pub fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu> {
        match &self.strategy {
            FusionStrategy::WeightedSum => self.weighted_sum.as_ref().unwrap().fuse(outputs, weights),
            FusionStrategy::ConcatLinear { .. } => self.concat_linear.as_ref().unwrap().fuse(outputs, weights),
            FusionStrategy::CrossAttention { .. } => self.cross_attention.as_ref().unwrap().fuse(outputs, weights),
        }
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        if let Some(cl) = &self.concat_linear { all.extend(cl.parameters()); }
        if let Some(ca) = &self.cross_attention { all.extend(ca.parameters()); }
        all
    }

    pub fn fused_dim(&self) -> usize { self.fused_dim }
    pub fn train(&mut self) { self.training = true; set_training_mode(true); }
    pub fn eval(&mut self)  { self.training = false; set_training_mode(false); }
}
```

---

# 四、src/weighted_sum.rs

```rust
//! WeightedSum：Σ gating_weight[i] × expert_output[i]
//! 无可训练参数，纯 Vec 逐元素累加（不碰算子）

use oyta_tensor::prelude::*;

pub struct WeightedSumFusion { expert_dim: usize }

impl WeightedSumFusion {
    pub fn new(expert_dim: usize) -> Self { Self { expert_dim } }

    pub fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu> {
        let num = outputs.len();
        if num == 0 {
            return Tensor::<f32, Cpu>::zeros([1, self.expert_dim]);
        }
        assert_eq!(weights.len(), num);
        let first_shape = outputs[0].1.shape().dims.clone();
        let total = first_shape.iter().product::<usize>();
        let mut fused = vec![0.0f32; total];
        for (i, (_, expert_out)) in outputs.iter().enumerate() {
            let w = weights[i];
            let d = expert_out.data_slice();
            for j in 0..total { fused[j] += w * d[j]; }
        }
        Tensor::<f32, Cpu>::from_data(fused, first_shape).unwrap()
    }
}
```

---

# 五、src/concat_linear.rs（Linear 调用 oyta_tensor）

```rust
//! ConcatLinear：concat 激活专家输出 → Linear 投影
//!
//! Linear 内部全调 oyta_tensor（matmul / add）

use oyta_tensor::prelude::*;
use oyta_expert::Linear;
use oyta_router::Parameter;

pub struct ConcatLinearFusion {
    expert_dim: usize,
    max_active: usize,
    proj_dim: usize,
    linear: Linear,  // Linear::forward 内部用 matmul + add
}

impl ConcatLinearFusion {
    pub fn new(did: u32, expert_dim: usize, proj_dim: usize) -> Self {
        let max_active = 4;
        Self {
            expert_dim, max_active, proj_dim,
            linear: Linear::new(did, "fusion.proj", max_active * expert_dim, proj_dim),
        }
    }

    pub fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], _weights: &[f32]) -> Tensor<f32, Cpu> {
        // (1) concat + padding
        let batch = outputs[0].1.shape().dims[0];
        let mut data: Vec<f32> = Vec::with_capacity(batch * self.max_active * self.expert_dim);
        for n in 0..batch {
            for (_, e) in outputs {
                let row = &e.data_slice()[n * self.expert_dim .. (n+1) * self.expert_dim];
                data.extend_from_slice(row);
            }
            for _ in 0..((self.max_active - outputs.len()) * self.expert_dim) { data.push(0.0); }
        }
        let concatted = Tensor::<f32, Cpu>::from_data(data, [batch, self.max_active * self.expert_dim]).unwrap();
        // (2) Linear 投影（内部调 oyta_tensor matmul + add）
        self.linear.forward(&concatted)
    }

    pub fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
}
```

---

# 六、src/task_head.rs

```rust
//! TaskHead：任务输出头
//!
//! ClassificationHead → Linear → logits（Softmax 在 Loss 内部做）
//! 全部算子走 oyta_tensor

use oyta_tensor::prelude::*;
use oyta_expert::Linear;
use oyta_router::Parameter;

pub trait TaskHeadBase: Send + Sync {
    fn forward(&self, fused: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;
    fn parameters(&self) -> Vec<Parameter>;
    fn output_dim(&self) -> usize;
}

pub enum TaskHead { Classification(ClassificationHead), Regression(RegressionHead) }

impl TaskHeadBase for TaskHead {
    fn forward(&self, f: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        match self { TaskHead::Classification(h) => h.forward(f), TaskHead::Regression(h) => h.forward(f) }
    }
    fn parameters(&self) -> Vec<Parameter> {
        match self { TaskHead::Classification(h) => h.parameters(), TaskHead::Regression(h) => h.parameters() }
    }
    fn output_dim(&self) -> usize {
        match self { TaskHead::Classification(h) => h.output_dim(), TaskHead::Regression(h) => h.output_dim() }
    }
}

pub struct ClassificationHead { linear: Linear, num_classes: usize }
impl ClassificationHead {
    pub fn new(did: u32, fused_dim: usize, num_classes: usize) -> Self {
        Self { linear: Linear::new(did, "task_head", fused_dim, num_classes), num_classes }
    }
}
impl TaskHeadBase for ClassificationHead {
    fn forward(&self, f: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> { self.linear.forward(f) }
    fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
    fn output_dim(&self) -> usize { self.num_classes }
}

pub struct RegressionHead { linear: Linear }
impl RegressionHead {
    pub fn new(did: u32, fused_dim: usize) -> Self {
        Self { linear: Linear::new(did, "task_head", fused_dim, 1) }
    }
}
impl TaskHeadBase for RegressionHead {
    fn forward(&self, f: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> { self.linear.forward(f) }
    fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
    fn output_dim(&self) -> usize { 1 }
}
```

---

# 七、oyta-fusion 验收标准

```
[ ] cargo check --package oyta-fusion 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1"（无 rust-version）
[ ] Linear 内部用 Tensor::matmul + Tensor::add（正确调用 oyta_tensor）
[ ] 每个函数有中文注释
```

---

# 八、四层分层目录规范（对齐 00-overview §八）

oyta-fusion 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层。核心：融合策略枚举 + FusionStrategy trait 归入 traits 层，三种具体融合实现归入 impl 层，Linear 投影底层复用 oyta-expert 的 utils。

## 8.1 目标目录结构

```
oyta-fusion/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude（只声明+导出）
│
├── types/                       ← 数据类型层（纯结构体定义）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   └── fusion_config.rs        ← FusionStrategy 枚举 + TaskHead 配置结构体
│
├── traits/                      ← 接口层（只 trait 定义）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   ├── fusion_strategy.rs      ← FusionStrategy trait { fuse(outputs, weights) → fused_tensor + fused_dim }
│   └── task_head.rs            ← TaskHeadBase trait { forward(fused) → logits + parameters + output_dim }
│
├── utils/                       ← 工具层（被 impl 层依赖）
│   ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
│   └── tensor_concat.rs        ← Tensor::concat 高层包装（支持任意 dim 拼接）
│
└── impl/                        ← 实现层（trait 的具体实现）
    ├── mod.rs                  ← 只 pub mod xxx;（零 pub use，零实现）
    ├── feature_fusion.rs       ← FeatureFusion 主类（策略分发 + 参数收集 + train/eval）
    ├── weighted_sum.rs          ← WeightedSumFusion（纯 Vec 加权累加，无可训练参数）
    ├── concat_linear.rs         ← ConcatLinearFusion（concat → Linear 投影，Linear 复用 oyta-expert）
    ├── cross_attention.rs       ← CrossAttentionFusion（Q=输出集合, K=V=gating权重向量 → FlashAttention + Linear）
    └── task_heads.rs           ← ClassificationHead + RegressionHead（Linear 复用 oyta-expert）
```

## 8.2 CrossAttentionFusion 关键设计（0.6.8 稀疏数据流）

```rust
//! CrossAttentionFusion 的融合语义（必须在 trait 或 impl 层体现）
//!
//! Query: 一个占位 tensor [batch, 1, fused_dim] — 类似 decoder 里的 CLS token
//! Key/Value: 所有激活专家输出 [batch, num_active, expert_dim]
//!   → 每个专家输出作为 K/V 的一行
//! Attention weight 由 Router 的 normalized_weights 作为 bias 加到 logits 上
//! → 这样既让专家自身的内容参与融合，又让 Router 的 gating 权重显式影响注意力

// 伪代码（指导实现方向）
let q = query_proj.forward(&cls_token);            // [batch, 1, d_model]
let kv = stack_expert_outputs(&expert_outs);        // [batch, num_active, expert_dim]
let kv = kv_proj.forward(&kv);                       // [batch, num_active, d_model]
let attn_out = flash_attention_multihead(            // 直接调 oyta_ops
    &q, &kv, &kv, num_heads, causal=false, scale,
);
let fused = self.out_proj.forward(&attn_out);       // [batch, 1, out_dim] → squeeze → [batch, out_dim]
```

## 8.3 ConcatLinearFusion 任意 dim 支持

```rust
//! utils/tensor_concat.rs 提供 concat_any_dim
//! 支持在任意 dim 上拼接（2D Tensor 拼接 dim=0 或 dim=1 都是合法场景）

pub fn concat_tensors(tensors: &[&Tensor<f32, Cpu>], dim: usize) -> Tensor<f32, Cpu> {
    // 使用 oyta_tensor 底层 Backend 的 concat（或手动组数据）
    // ConcatLinearFusion 默认 dim=1（特征维拼接），调用时可指定
    // 例：[batch, d1] concat [batch, d2] at dim=1 → [batch, d1+d2]
    unimplemented!("Phase 0.5 补实现")
}
```

## 8.4 每层职责边界 + mod.rs 铁律（严格对齐 00-overview §八）

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + traits/ + utils/ + oyta_expert | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! traits 层说明

pub mod fusion_strategy;
pub mod task_head;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::fusion_config::FusionStrategy;
pub use traits::fusion_strategy::FusionStrategyTrait;
pub use traits::task_head::TaskHeadBase;
pub use utils::tensor_concat::concat_tensors;
pub use impls::feature_fusion::FeatureFusion;
pub use impls::weighted_sum::WeightedSumFusion;
pub use impls::concat_linear::ConcatLinearFusion;
pub use impls::cross_attention::CrossAttentionFusion;
pub use impls::task_heads::{ClassificationHead, RegressionHead};
```

---

# 九、验收标准（扩充版：含 CrossAttention + Concat 任意 dim）

```
## 基础验收（原有）
[ ] cargo check --package oyta-fusion 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1" + oyta-expert（workspace path，无 rust-version）
[ ] Linear 内部用 Tensor::matmul + Tensor::add（正确调用 oyta_tensor）
[ ] 每个函数有中文注释

## 追加：CrossAttentionFusion 验收
[ ] CrossAttentionFusion 存在且实现 FeatureFusion trait
[ ] 内部正确调用 oyta_tensor::flash_attn::flash_attention_multihead（或单头 flash_attention）
[ ] Query = 占位 CLS token [batch, 1, d_model]
[ ] Key/Value = 所有激活专家输出堆叠 [batch, num_active, expert_dim]
[ ] 可选：gating 权重作为 attention logits 的 bias（让 Router 显式影响融合）
[ ] out_proj Linear 投影回 fused_dim

## 追加：ConcatLinearFusion 验收（任意 dim）
[ ] concat_linear.rs 支持 dim 参数（默认 dim=1 特征维拼接）
[ ] utils/tensor_concat.rs 提供 concat_tensors(tensors, dim) → Tensor
[ ] 2D Tensor [batch, d1] concat [batch, d2] at dim=1 → [batch, d1+d2] ✓
[ ] 2D Tensor [batch1, d] concat [batch2, d] at dim=0 → [batch1+batch2, d] ✓
[ ] 3D Tensor 支持 dim=0/1/2 任意拼接（对应不同的专家输出维度布局）

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use（types::fusion_config::FusionStrategy 等）
[ ] 依赖方向单向：types → traits/utils → impls
[ ] FusionStrategy 枚举归入 types/ 层
[ ] FusionStrategyTrait + TaskHeadBase trait 归入 traits/ 层
[ ] concat_tensors 归入 utils/ 层
[ ] FeatureFusion + WeightedSumFusion + ConcatLinearFusion + CrossAttentionFusion + ClassificationHead + RegressionHead 归入 impl/ 层
```
