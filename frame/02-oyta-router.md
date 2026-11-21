# oyta-router 开发文档（Phase 2）

> crate：`oyta-router/` | 依赖：**`oyta-tensor = "0.1.1"` + `oyta-content`** | 职责：稀疏路由策略 + 三层掩码 + 按领域独立的 SparseParameterPool

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-router"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
oyta-content = { version = "0.1.0", path = "../oyta-content" }
serde       = { version = "1.0.229", features = ["derive"] }
serde_json  = "1.0.152"
rayon       = "1.12.0"
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs

```rust
pub mod routing_strategy;
pub mod sparse_mask;
pub mod param;
pub mod param_pool;
pub mod domain_router;
pub mod op_type;

pub use routing_strategy::{RoutingStrategy, RoutingResult};
pub use sparse_mask::{SparseMaskSet, DomainSparseMask};
pub use param::Parameter;
pub use param_pool::SparseParameterPool;
pub use domain_router::DomainRouter;
pub use op_type::OpType;

pub mod prelude {
    pub use crate::{
        RoutingStrategy, RoutingResult, SparseMaskSet, DomainSparseMask,
        Parameter, SparseParameterPool, DomainRouter, OpType,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_content::ContentAnalysis;
}
```

---

# 三、src/routing_strategy.rs

```rust
//! RoutingStrategy：稀疏路由策略（纯算法）
//!
//! 两种策略：TopK / Threshold，都带 softmax 归一化 + 空集 fallback

use std::collections::HashSet;

/// 稀疏路由策略枚举
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    TopK { k: usize },
    Threshold { threshold: f32 },
}

/// 路由结果（route() 一次返回的完整决策）
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub mask: Vec<bool>,                    // len = num_domains，true = 激活
    pub active_indices: Vec<u32>,            // 按权重降序的激活领域 ID
    pub normalized_weights: Vec<f32>,        // softmax 归一化，和 = 1.0
}

impl RoutingStrategy {
    pub fn top_k(k: usize) -> Self { Self::TopK { k } }
    pub fn threshold(t: f32) -> Self { Self::Threshold { threshold: t } }

    /// 主入口：route() 一次返回完整决策
    pub fn route(&self, weights: &[(u32, f32)], num_domains: usize) -> RoutingResult {
        let mask = self.generate_domain_mask(weights, num_domains);
        let mut active_indices: Vec<u32> = Vec::new();
        for &(id, _) in weights {
            if mask.get(id as usize).copied().unwrap_or(false) {
                active_indices.push(id);
            }
        }
        let (_, nw) = self.compute_normalized_weights(weights, &active_indices);
        RoutingResult { mask, active_indices, normalized_weights: nw }
    }

    /// 生成领域级 mask（L1 来源）
    pub fn generate_domain_mask(&self, weights: &[(u32, f32)], num_domains: usize) -> Vec<bool> {
        let mut mask = vec![false; num_domains];
        match self {
            Self::TopK { k } => {
                let mut sorted: Vec<(u32, f32)> = weights.iter().cloned().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
                for i in 0..(*k).min(sorted.len()) {
                    let did = sorted[i].0 as usize;
                    if did < num_domains { mask[did] = true; }
                }
            }
            Self::Threshold { threshold } => {
                for &(did, w) in weights {
                    if w > *threshold && (did as usize) < num_domains {
                        mask[did as usize] = true;
                    }
                }
            }
        }
        // 空集 fallback（至少激活权重最高的那个）
        if !mask.iter().any(|&m| m) {
            let best = weights.iter().cloned()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(id, _)| id as usize)
                .unwrap_or(0);
            if best < num_domains { mask[best] = true; }
        }
        mask
    }

    /// softmax 归一化（max 减式防溢出）
    pub fn compute_normalized_weights(
        &self,
        weights: &[(u32, f32)],
        active_indices: &[u32],
    ) -> (Vec<u32>, Vec<f32>) {
        let id_set: HashSet<u32> = active_indices.iter().copied().collect();
        let active_raw: Vec<(u32, f32)> = weights.iter()
            .filter(|(id, _)| id_set.contains(id))
            .cloned()
            .collect();
        if active_raw.is_empty() { return (Vec::new(), Vec::new()); }
        let max_w = active_raw.iter().map(|(_, w)| *w).fold(f32::NEG_INFINITY, f32::max);
        let max_w = if max_w.is_finite() { max_w } else { 0.0 };
        let exp_vec: Vec<f32> = active_raw.iter().map(|(_, w)| (w - max_w).exp()).collect();
        let sum_exp: f32 = exp_vec.iter().sum();
        let sum_exp = if sum_exp > 0.0 { sum_exp } else { 1.0 };
        let norm: Vec<f32> = exp_vec.iter().map(|e| e / sum_exp).collect();
        let ids: Vec<u32> = active_raw.iter().map(|(id, _)| *id).collect();
        (ids, norm)
    }
}
```

---

# 四、src/sparse_mask.rs

```rust
//! SparseMaskSet：三层掩码全集
//!
//! L1 领域级（domain_mask: Vec<bool>）
//! L2 通道级（channel_masks: HashMap<u32, Vec<f32>>）  mask[f] ∈ [0,1]
//! L3 序列级（sequence_masks: HashMap<(u32, usize), Vec<f32>>）  mask[t] ∈ [0,1]

use std::collections::HashMap;

/// 单个领域的 Op 消费掩码
#[derive(Debug, Clone, Default)]
pub struct DomainSparseMask {
    pub channel_mask: Option<Vec<f32>>,    // L2
    pub sequence_mask: Option<Vec<f32>>,   // L3
}

/// 完整三层掩码集（一个 batch 的全部稀疏决策）
#[derive(Debug, Clone)]
pub struct SparseMaskSet {
    pub domain_mask: Vec<bool>,
    pub channel_masks: HashMap<u32, Vec<f32>>,
    pub sequence_masks: HashMap<(u32, usize), Vec<f32>>,
}

impl SparseMaskSet {
    pub fn new(num_domains: usize) -> Self {
        Self { domain_mask: vec![false; num_domains],
               channel_masks: HashMap::new(), sequence_masks: HashMap::new() }
    }

    // ===== L1 =====
    pub fn set_domain_active(&mut self, did: u32, active: bool) {
        let i = did as usize;
        if i < self.domain_mask.len() { self.domain_mask[i] = active; }
    }
    pub fn is_active(&self, did: u32) -> bool { self.domain_mask.get(did as usize).copied().unwrap_or(false) }
    pub fn active_ids(&self) -> Vec<u32> {
        self.domain_mask.iter().enumerate().filter(|(_,&m)| m).map(|(i,_)| i as u32).collect()
    }

    // ===== L2 =====
    pub fn set_channel_mask(&mut self, did: u32, mask: Vec<f32>) { self.channel_masks.insert(did, mask); }
    pub fn get_channel_mask(&self, did: u32) -> Option<&Vec<f32>> { self.channel_masks.get(&did) }

    // ===== L3 =====
    pub fn set_sequence_mask(&mut self, did: u32, channel: usize, mask: Vec<f32>) {
        self.sequence_masks.insert((did, channel), mask);
    }
    pub fn get_sequence_mask(&self, did: u32) -> Option<&Vec<f32>> { self.sequence_masks.get(&(did, 0)) }

    // ===== 工厂方法 =====
    pub fn build(
        weights: &[(u32, f32)],
        num_domains: usize,
        strategy: &RoutingStrategy,
        preset_channels: &HashMap<u32, Option<Vec<f32>>>,
        preset_sequences: &HashMap<u32, Option<Vec<f32>>>,
    ) -> Self {
        let mut mask_set = SparseMaskSet::new(num_domains);
        let dm = strategy.generate_domain_mask(weights, num_domains);
        for (i, &a) in dm.iter().enumerate() { mask_set.set_domain_active(i as u32, a); }
        for did in 0..num_domains {
            let did_u32 = did as u32;
            if !mask_set.is_active(did_u32) { continue; }
            if let Some(Some(cm)) = preset_channels.get(&did_u32) { mask_set.set_channel_mask(did_u32, cm.clone()); }
            if let Some(Some(sm)) = preset_sequences.get(&did_u32) { mask_set.set_sequence_mask(did_u32, 0, sm.clone()); }
        }
        mask_set
    }
}
```

---

# 五、src/param.rs

```rust
//! Parameter：单个可训练参数包装
//!
//! data 用 oyta_tensor::Tensor<f32, Cpu>（不是自己手搓）
//! Arc<Mutex> 让 Parameter 可 clone + 多线程安全

use std::sync::{Arc, Mutex};
use oyta_tensor::prelude::*;

/// 可训练参数
pub struct Parameter {
    /// 参数值（Arc<Mutex<Tensor<f32, Cpu>>>）
    pub data: Arc<Mutex<Tensor<f32, Cpu>>>,
    /// 梯度（None = 还没 backward 或已 zero_grad）
    pub grad: Option<Tensor<f32, Cpu>>,
    /// 是否需要梯度（冻结参数 = false）
    pub requires_grad: bool,
    /// 命名格式：domain_{did}.{module}.{param}（序列化用）
    pub name: String,
}

impl Parameter {
    /// 构造（用 oyta_tensor::Tensor）
    pub fn new(data: Tensor<f32, Cpu>, name: impl Into<String>) -> Self {
        Self { data: Arc::new(Mutex::new(data)), grad: None, requires_grad: true, name: name.into() }
    }
    pub fn freeze(mut self) -> Self { self.requires_grad = false; self }

    pub fn zero_grad(&mut self) {
        if self.requires_grad { self.grad = None; }
    }
    pub fn has_grad(&self) -> bool { self.grad.is_some() }
    pub fn grad(&self) -> Option<&Tensor<f32, Cpu>> { self.grad.as_ref() }
    pub fn grad_mut(&mut self) -> Option<&mut Tensor<f32, Cpu>> { self.grad.as_mut() }
    pub fn set_grad(&mut self, g: Tensor<f32, Cpu>) {
        if self.requires_grad { self.grad = Some(g); }
    }
    /// 梯度累加（梯度累积跨 batch 用）
    pub fn accumulate_grad(&mut self, new_grad: &Tensor<f32, Cpu>) {
        if !self.requires_grad { return; }
        match &mut self.grad {
            Some(existing) => {
                let ed = existing.data_slice(); let nd = new_grad.data_slice();
                let sum: Vec<f32> = ed.iter().zip(nd.iter()).map(|(a,b)| a+b).collect();
                let shape = existing.shape().dims.clone();
                *existing = Tensor::<f32, Cpu>::from_data(sum, shape).expect("grad 累加构造失败");
            }
            None => { self.grad = Some(new_grad.clone()); }
        }
    }
    pub fn numel(&self) -> usize { self.data.lock().unwrap().numel() }
}

/// 按命名约定构造 Parameter（框架层统一入口）
pub fn build_param(domain_id: u32, module: &str, name: &str, data: Tensor<f32, Cpu>) -> Parameter {
    let full = format!("domain_{}.{}.{}", domain_id, module, name);
    Parameter::new(data, full)
}
```

---

# 六、src/param_pool.rs

```rust
//! SparseParameterPool：参数按领域独立存储
//!
//! zero_grad_sparse / step_sparse 只遍历 active_ids 内领域
//! 复杂度：O(active_params) << O(all_params)

use std::collections::HashMap;
use crate::param::Parameter;

pub struct SparseParameterPool {
    domain_params: HashMap<u32, Vec<Parameter>>,
    order: Vec<u32>,  // 稳定遍历顺序
    total: usize,     // 总参数量
}

impl SparseParameterPool {
    pub fn new() -> Self { Self { domain_params: HashMap::new(), order: Vec::new(), total: 0 } }

    pub fn register_domain(&mut self, did: u32, params: Vec<Parameter>) {
        let n: usize = params.iter().map(|p| p.numel()).sum();
        self.domain_params.entry(did)
            .and_modify(|v| v.extend(params.clone()))
            .or_insert_with(|| { self.order.push(did); params });
        self.total += n;
    }

    pub fn domain_params(&self, did: u32) -> &[Parameter] {
        self.domain_params.get(&did).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn domain_params_mut(&mut self, did: u32) -> &mut Vec<Parameter> {
        self.domain_params.entry(did).or_insert_with(|| { self.order.push(did); Vec::new() })
    }
    pub fn registered_ids(&self) -> &[u32] { &self.order }
    pub fn total_params(&self) -> usize { self.total }

    /// 稀疏清零（只清激活领域参数的 grad）
    pub fn zero_grad_sparse(&mut self, active: &[u32]) {
        for &did in active {
            if let Some(ps) = self.domain_params.get_mut(&did) {
                for p in ps { p.zero_grad(); }
            }
        }
    }

    /// 稀疏 step（只更新激活领域参数的值，闭包由 Optimizer 提供）
    pub fn step_sparse<F>(&mut self, active: &[u32], mut f: F)
    where F: FnMut(&mut Parameter) {
        for &did in active {
            if let Some(ps) = self.domain_params.get_mut(&did) {
                for p in ps { f(p); }
            }
        }
    }

    pub fn count_active_params(&self, active: &[u32]) -> usize {
        active.iter().filter_map(|did| self.domain_params.get(did))
            .flat_map(|ps| ps.iter()).map(|p| p.numel()).sum()
    }
}

impl Default for SparseParameterPool { fn default() -> Self { Self::new() } }
```

---

# 七、src/domain_router.rs

```rust
//! DomainRouter：组装 Strategy + Mask + Pool
//! 对外暴露 route() 方法（每个 batch 动态决策）

use crate::routing_strategy::{RoutingStrategy, RoutingResult};
use crate::sparse_mask::SparseMaskSet;
use crate::param_pool::SparseParameterPool;
use oyta_content::ContentAnalysis;
use std::collections::HashMap;

pub struct DomainRouter {
    strategy: RoutingStrategy,
    num_domains: usize,
    preset_channels: HashMap<u32, Option<Vec<f32>>>,
    preset_sequences: HashMap<u32, Option<Vec<f32>>>,
    pub param_pool: SparseParameterPool,
}

impl DomainRouter {
    pub fn new(strategy: RoutingStrategy, num_domains: usize) -> Self {
        Self { strategy, num_domains,
               preset_channels: HashMap::new(), preset_sequences: HashMap::new(),
               param_pool: SparseParameterPool::new() }
    }
    pub fn with_channel_mask(mut self, did: u32, mask: Vec<f32>) -> Self {
        self.preset_channels.insert(did, Some(mask)); self
    }
    pub fn num_domains(&self) -> usize { self.num_domains }

    /// 核心方法：一个 batch 的完整决策
    /// 返回 (RoutingResult, SparseMaskSet)
    pub fn route(&self, analysis: &ContentAnalysis) -> (RoutingResult, SparseMaskSet) {
        let weights = &analysis.domain_weights;
        let result = self.strategy.route(weights, self.num_domains);

        // 注入 ContentAnalyzer 检测到的 padding mask（L3）
        let mut seq_input = self.preset_sequences.clone();
        if let Some(ref pm) = analysis.padding_mask {
            if let Some(first) = pm.first() {
                for did in &result.active_indices {
                    seq_input.insert(*did, Some(first.clone()));
                }
            }
        }

        let mask_set = SparseMaskSet::build(
            weights, self.num_domains, &self.strategy,
            &self.preset_channels, &seq_input,
        );
        (result, mask_set)
    }
}
```

---

# 八、src/op_type.rs（给 autograd crate resolve_op_mask 用）

```rust
//! OpType：算子类型枚举（避免 router → autograd 循环依赖）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Conv1d, Conv1dMasked, Conv1dWithSequenceMask,
    BatchNorm1d, BatchNorm1dMasked,
    MaxPool1d, AvgPool1d, Linear, Matmul,
    Relu, Gelu, Sigmoid, Tanh, Silu,
    Softmax, LayerNorm, RMSNorm, FlashAttention,
}
```

---

# 九、稀疏掩码三层生成完整流程（架构 0.6.8 扩展）

## 9.1 MaskLevel 枚举显式定义

```rust
// oyta-router/src/sparse_mask.rs
//! 稀疏掩码层级（由粗到细，每个可以独立开关）
#[derive(Debug, Clone)]
pub enum MaskLevel {
    DomainMask,                                    // Level 1: 领域级（整个专家激活/跳过）
    ChannelMask { expert_domain_id: u32 },         // Level 2: 通道级（领域内特征选择/Early Exit）
    SequenceMask { expert_domain_id: u32, channel: usize }, // Level 3: 序列级（变长/padding）
}
```

## 9.2 SparseMaskSet::build 完整逻辑（补充文档 §四）

> 现有文档写了 SparseMaskSet 的字段，但 build() 的完整生成流程在架构 0.6.8 有详细步骤，这里补齐

```rust
impl SparseMaskSet {
    /// 从 ContentAnalysis.domain_weights 生成完整掩码集合
    /// 完整流程：
    ///   1. RoutingStrategy 生成 L1 domain_mask
    ///   2. 注入预设 L2 channel_mask（如果配置了）
    ///   3. 注入 ContentAnalyzer 检测到的 L3 padding_mask
    pub fn build(weights, total_domains, strategy, preset_channels, preset_sequences) -> Self {
        // Step 1: L1 —— generate_domain_mask（TopK: 降序 → 前 K; Threshold: 权重>阈值）
        //          保证至少激活一个（空集 fallback → 激活权重最高的那个领域）
        let domain_mask = strategy.generate_domain_mask(weights, total_domains);

        // Step 2: L2 —— 只给激活专家应用预设 channel_mask
        let mut channel_masks = HashMap::new();
        for (did, mask) in preset_channels {
            if domain_mask.get(*did as usize) == Some(&true) {
                channel_masks.insert(*did, mask.clone().unwrap_or_default());
            }
        }

        // Step 3: L3 —— 只给激活专家应用 sequence_mask（padding mask 或 preset）
        let mut sequence_masks = HashMap::new();
        for (did, mask_opt) in preset_sequences {
            if domain_mask.get(*did as usize) == Some(&true) {
                if let Some(mask) = mask_opt {
                    // 给每个通道都注入同一份 sequence_mask
                    // （卷积层所有输出通道共享 padding mask）
                    for ch in 0..num_channels_hint {
                        sequence_masks.insert((*did, ch), mask.clone());
                    }
                }
            }
        }

        Self { domain_mask, channel_masks, sequence_masks }
    }
}
```

## 9.3 RoutingStrategy 两个核心方法（架构 0.6.8 补充）

> 现有文档 §二 写了 route()，但架构 0.6.8 还有 generate_domain_mask() 和 compute_normalized_weights()

```rust
impl RoutingStrategy {
    /// 生成 Level 1 domain_mask（返回 Vec<bool>，len=num_domains）
    /// DomainRouter.route() 内部调用
    pub fn generate_domain_mask(&self, domain_weights: &[(u32, f32)], total_domains: usize) -> Vec<bool> {
        let mut mask = vec![false; total_domains];
        match self {
            TopK { k } => {
                let mut sorted: Vec<(u32, f32)> = domain_weights.iter().cloned().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..(*k).min(sorted.len()) {
                    if let Some(&(did, _)) = sorted.get(i) { mask[did as usize] = true; }
                }
            }
            Threshold { threshold } => {
                for &(did, w) in domain_weights { if w > *threshold { mask[did as usize] = true; } }
            }
        }
        // ===== 空集 fallback：保证至少激活一个 =====
        if !mask.iter().any(|&m| m) {
            if let Some(&(best_id, _)) = domain_weights.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) {
                mask[best_id as usize] = true;
            }
        }
        mask
    }

    /// 同时返回归一化权重（FeatureFusion gating 用）
    pub fn compute_normalized_weights(&self, domain_weights: &[(u32, f32)]) -> (Vec<u32>, Vec<f32>) {
        // 1. 确定 active_ids（复用 generate_domain_mask 的逻辑）
        // 2. 取激活领域原始权重 → softmax 归一化
        // 3. 返回 (active_ids, normalized_weights)
    }
}
```

---

# 十、与 oyta-autograd 接口约定（架构 0.6.8）

## 10.1 OpType 枚举（已在 §八，但补充掩码变体字段说明）

```rust
// oyta-router/src/op_type.rs
//! OpType：算子类型枚举（避免 router → autograd 循环依赖）
//!
//! 三个掩码变体的含义：
//!   Conv1dMasked          —— 带通道级 mask（某些输出通道被 mask 掉）
//!   Conv1dWithSequenceMask —— 带序列级 mask（变长/padding 位置不参与卷积）
//!   BatchNorm1dMasked      —— 带通道级 mask（某些特征跳过 BN 归一化）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Conv1d, Conv1dMasked, Conv1dWithSequenceMask,
    BatchNorm1d, BatchNorm1dMasked,
    MaxPool1d, AvgPool1d, Linear, Matmul,
    Relu, Gelu, Sigmoid, Tanh, Silu,
    Softmax, LayerNorm, RMSNorm, FlashAttention,
}
```

## 10.2 算子内部 mask 应用逻辑（backward 时乘 (1-mask)）

> 这部分逻辑在 oyta-autograd 的 `compute_backward()` 里，但 router 需要知道 mask 如何传递

```rust
// oyta-autograd/src/op.rs compute_backward() 关键分支
// Op::Conv1dMasked backward 时：
//   grad_out_masked[batch, out_c, out_len] *= (1 - channel_mask[out_c])
//   即 mask=1 的输出通道梯度清零（forward 时输出就是 0）
//   然后调标准 Conv1d backward

// Op::BatchNorm1dMasked backward 时：
//   grad_x[batch, num_features] 的 feature f 如果被 mask → 清零
//   grad_gamma[num_features] 的 feature f 如果被 mask → 清零
//   grad_beta 不被 mask（bias 梯度无意义清零）
```

## 10.3 DomainRouter.route() 返回值格式

```rust
pub fn route(&self, analysis: &ContentAnalysis) -> (RoutingResult, SparseMaskSet)
// RoutingResult:
//   pub active_indices: Vec<u32>    // 激活的领域 ID（FeatureFusion 用）
//   pub active_weights: Vec<f32>    // 对应归一化权重（gating 用）
// SparseMaskSet:
//   pub domain_mask: Vec<bool>
//   pub channel_masks: HashMap<u32, Vec<f32>>
//   pub sequence_masks: HashMap<(u32, usize), Vec<f32>>
```

---

# 十一、四层分层目录规范（对齐 00-overview §八）

> 现有 02 文档 §二 lib.rs 写的是平铺 src/，这里更正为四层分层

```
oyta-router/src/
├── lib.rs                              ← 顶层统一 pub mod + pub use + prelude
├── types/
│   ├── mod.rs                          ← 只 pub mod 声明
│   ├── sparse_mask.rs                  ← SparseMaskSet + MaskLevel + SparseMaskSet::build
│   └── op_type.rs                      ← OpType 枚举（避免 router → autograd 循环依赖）
├── traits/                             ← router crate 可能没有独立 trait 层
│   └── mod.rs
├── utils/
│   ├── mod.rs                          ← 只 pub mod 声明
│   ├── param.rs                        ← Parameter + build_param（domain_{did}.module.param 命名）
│   └── param_pool.rs                   ← SparseParameterPool + zero_grad_sparse / step_sparse
└── impl/ (或直接 analyzers/)
    ├── mod.rs                          ← 只 pub mod 声明
    ├── routing_strategy.rs            ← RoutingStrategy（TopK / Threshold）
    └── domain_router.rs                ← DomainRouter（route 主入口）
```

---

# 十二、oyta-router 验收标准

```
[ ] cargo check --package oyta-router 零警告
[ ] Cargo.toml 无 path 依赖、无 rust-version，oyta-tensor = "0.1.1" + oyta-content = workspace path
[ ] Parameter.data 用 Arc<Mutex<Tensor<f32, Cpu>>>（正确调用 oyta-tensor）
[ ] Parameter.name 格式 = "domain_{did}.{module}.{param}"（序列化用）
[ ] RoutingStrategy::TopK(2) 对 [(0,0.05),(1,0.85),(2,0.03),(3,0.07)] → active=[1,3]
[ ] 空集 fallback：所有权重=0 → 激活权重最高领域（不返回空 active 集）
[ ] SparseParameterPool::zero_grad_sparse([1,0]) 只清 domain_id=1 和 0 的参数
[ ] SparseParameterPool::step_sparse([1,0]) 闭包只对 domain_id=1 和 0 执行
[ ] SparseMaskSet::build 正确注入 padding_mask → 只有激活专家有 L3 mask
[ ] 每个函数有中文注释
[ ] mod.rs 只做 pub mod 声明
```
