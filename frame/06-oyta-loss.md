# oyta-loss 开发文档（Phase 6）

> crate：`oyta-loss/` | 依赖：**`oyta-tensor = "0.1.1"`** | 职责：损失函数集 + MultiDomainLoss

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-loss"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs + src/loss_fn.rs + src/reduction.rs

```rust
// src/lib.rs
pub mod loss_fn;
pub mod reduction;
pub mod cross_entropy;
pub mod mse;
pub mod multi_domain;

pub use loss_fn::LossFn;
pub use reduction::Reduction;
pub use cross_entropy::CrossEntropyLoss;
pub use mse::MSELoss;
pub use multi_domain::MultiDomainLoss;

pub mod prelude {
    pub use crate::{LossFn, Reduction, CrossEntropyLoss, MSELoss, MultiDomainLoss};
    pub use oyta_tensor::prelude::*;
}

// src/reduction.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction { Mean, Sum, None }

// src/loss_fn.rs
use oyta_tensor::prelude::*;
use crate::reduction::Reduction;

pub trait LossFn: Send + Sync {
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;
    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;
    fn item(&self, loss: &Tensor<f32, Cpu>) -> f32 {
        if loss.numel() == 1 { return loss.data_slice()[0]; }
        let d = loss.data_slice();
        d.iter().sum::<f32>() / d.len() as f32
    }
    fn reduction(&self) -> Reduction;
}
```

---

# 三、src/cross_entropy.rs

```rust
//! CrossEntropyLoss：Softmax + NLL（log_softmax 数值稳定版本）
//!
//! grad = softmax(logits) - one_hot(target)

use oyta_tensor::prelude::*;
use crate::loss_fn::LossFn;
use crate::reduction::Reduction;

pub struct CrossEntropyLoss {
    pub num_classes: usize,
    pub reduction: Reduction,
    pub label_smoothing: f32,
}

impl CrossEntropyLoss {
    pub fn new(num_classes: usize, reduction: Reduction) -> Self {
        Self { num_classes, reduction, label_smoothing: 0.0 }
    }
}

impl LossFn for CrossEntropyLoss {
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0];
        let nc = self.num_classes;
        let ld = logits.data_slice(); let td = target.data_slice();

        let mut per_sample: Vec<f32> = Vec::with_capacity(batch);
        for n in 0..batch {
            let row = n * nc;
            let max_v = (0..nc).map(|c| ld[row + c]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for c in 0..nc { sum_exp += (ld[row + c] - max_v).exp() as f64; }
            let log_sum_exp = sum_exp.ln() as f32 + max_v;
            let mut loss_nll = 0.0f64;
            let eps = self.label_smoothing;
            if target.numel() == batch {
                let class = td[n] as usize;
                loss_nll = -(1.0 - eps) * (ld[row + class] - log_sum_exp) as f64;
                for c in 0..nc {
                    if c != class { loss_nll -= (eps / (nc as f64 - 1.0).max(1.0)) * (ld[row + c] - log_sum_exp) as f64; }
                }
            } else {
                for c in 0..nc { loss_nll -= td[n * nc + c] as f64 * (ld[row + c] - log_sum_exp) as f64; }
            }
            per_sample.push(loss_nll as f32);
        }
        match self.reduction {
            Reduction::Mean => {
                let m = per_sample.iter().sum::<f32>() / per_sample.len() as f32;
                Tensor::<f32, Cpu>::from_data(vec![m], [1]).unwrap()
            }
            Reduction::Sum => {
                let s: f32 = per_sample.iter().sum();
                Tensor::<f32, Cpu>::from_data(vec![s], [1]).unwrap()
            }
            Reduction::None => Tensor::<f32, Cpu>::from_data(per_sample, [batch]).unwrap(),
        }
    }

    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0]; let nc = self.num_classes;
        let ld = logits.data_slice(); let td = target.data_slice();
        let mut grad = vec![0.0f32; batch * nc];

        for n in 0..batch {
            let row = n * nc;
            let max_v = (0..nc).map(|c| ld[row + c]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f64;
            for c in 0..nc { sum_exp += (ld[row + c] - max_v).exp() as f64; }
            let sm: Vec<f32> = (0..nc).map(|c| (ld[row + c] - max_v).exp() as f64 / sum_exp).collect::<Vec<f64>>().iter().map(|&v| v as f32).collect();

            let mut one_hot = vec![0.0f32; nc];
            if target.numel() == batch { one_hot[td[n] as usize] = 1.0; }
            else { for c in 0..nc { one_hot[c] = td[n * nc + c]; } }

            for c in 0..nc { grad[row + c] = sm[c] - one_hot[c]; }
        }
        if matches!(self.reduction, Reduction::Mean) { let scale = 1.0 / batch as f32; for g in grad.iter_mut() { *g *= scale; } }
        Tensor::<f32, Cpu>::from_data(grad, [batch, nc]).unwrap()
    }

    fn reduction(&self) -> Reduction { self.reduction }
}
```

---

# 四、src/multi_domain.rs

```rust
//! MultiDomainLoss：主损失 + 只给激活领域加辅助损失
//!
//! L = main(fused, target) + aux_weight × Σ_{did ∈ active} aux[did](expert_out[did], target)

use std::collections::HashMap;
use oyta_tensor::prelude::*;
use crate::loss_fn::LossFn;

pub struct MultiDomainLoss {
    pub main_loss: Box<dyn LossFn>,
    pub aux_losses: HashMap<u32, Box<dyn LossFn>>,
    pub aux_weight: f32,
}

impl MultiDomainLoss {
    pub fn new(main: Box<dyn LossFn>) -> Self {
        Self { main_loss: main, aux_losses: HashMap::new(), aux_weight: 0.1 }
    }

    pub fn with_auxiliary(&mut self, did: u32, loss: Box<dyn LossFn>) {
        self.aux_losses.insert(did, loss);
    }

    pub fn compute(
        &self,
        fused: &Tensor<f32, Cpu>,
        target: &Tensor<f32, Cpu>,
        active: &[u32],
        expert_outs: &[(u32, Tensor<f32, Cpu>)],
    ) -> Tensor<f32, Cpu> {
        let main_val = self.main_loss.item(&self.main_loss.compute(fused, target));
        let mut total = main_val;
        if self.aux_weight > 0.0 {
            for (did, out) in expert_outs {
                if !active.contains(did) { continue; }
                if let Some(aux) = self.aux_losses.get(did) {
                    total += self.aux_weight * aux.item(&aux.compute(out, target));
                }
            }
        }
        Tensor::<f32, Cpu>::from_data(vec![total], [1]).unwrap()
    }

    pub fn item(&self, loss: &Tensor<f32, Cpu>) -> f32 {
        if loss.numel() == 1 { loss.data_slice()[0] } else { loss.data_slice().iter().sum::<f32>() / loss.numel() as f32 }
    }
}
```

---

# 五、oyta-loss 验收标准

```
[ ] cargo check --package oyta-loss 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1"（无 rust-version）
[ ] CrossEntropyLoss.grad 对 logits=[[1,0,0],[0,1,0]] class=[0,1]
    → grad[0] = softmax - one_hot（正确）
[ ] MultiDomainLoss.compute 只对 active 里的领域加辅助损失
[ ] 每个函数有中文注释
```

---

# 六、四层分层目录规范（对齐 00-overview §八）

oyta-loss 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层。核心：LossFn trait 归入 traits 层，CrossEntropyLoss / MSELoss / MultiDomainLoss 归入 impl 层，Reduction 枚举归入 types 层。

## 6.1 目标目录结构

```
oyta-loss/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude
│
├── types/                       ← 数据类型层（纯枚举/结构体）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── reduction.rs            ← Reduction 枚举 { Mean, Sum, None }
│
├── traits/                      ← 接口层（只 trait 定义）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── loss_fn.rs              ← LossFn trait { compute, grad, item, reduction }
│
├── utils/                       ← 工具层（可复用数学组件）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── log_softmax.rs          ← 数值稳定 log_softmax（CrossEntropy 共享）
│
└── impl/                        ← 实现层（trait 的具体实现）
    ├── mod.rs                  ← 只 pub mod xxx;
    ├── cross_entropy.rs        ← CrossEntropyLoss { num_classes, reduction, label_smoothing }
    ├── mse.rs                  ← MSELoss
    └── multi_domain.rs         ← MultiDomainLoss（主损失 + active_ids 过滤辅助损失）
```

## 6.2 MultiDomainLoss 按 active 过滤辅助损失的精确语义

```rust
//! multi_domain.rs 中的关键过滤逻辑（架构 0.6.8）
//!
//! 问题：一个 batch 只激活 K 个领域 → 只有这 K 个专家有 expert_out 传进 compute
//!       未激活领域的专家 forward 被跳过 → 没有输出 → 不能对它们算辅助损失
//!
//! 正确做法：遍历 expert_outs（已经只包含激活领域的输出），对每个检查：
//!   (did, out) in expert_outs → 如果 did ∈ active 且存在 aux_losses[did] → 加辅助损失
//!
//! 绝对不能：
//!   ❌ 遍历所有 aux_losses.keys() 然后不管 active 都算（会对未激活领域算 loss，但梯度无法回传）
//!   ❌ 遍历所有 registered domain_ids（未激活领域没有 expert_out → forward 就跑了，稀疏白做）

pub fn compute(
    &self,
    fused: &Tensor<f32, Cpu>,
    target: &Tensor<f32, Cpu>,
    active: &[u32],                                    // Router 返回的激活领域列表
    expert_outs: &[(u32, Tensor<f32, Cpu>)],          // 只含激活领域的专家输出
) -> Tensor<f32, Cpu> {
    let main_val = self.main_loss.item(&self.main_loss.compute(fused, target));
    let mut total = main_val;

    if self.aux_weight > 0.0 {
        // 遍历 expert_outs（已过滤：只含激活领域）
        for (did, out) in expert_outs {
            // 二次确认：did ∈ active（防御性检查）
            if !active.contains(did) { continue; }
            if let Some(aux) = self.aux_losses.get(did) {
                total += self.aux_weight * aux.item(&aux.compute(out, target));
            }
        }
        // ⚠️ 未激活领域的 aux_loss 不参与总损失 → 它们的 Parameter 接收不到梯度
        // 这和 autograd sparse_backward + optim step_sparse 的过滤是一致的三层防线
    }

    Tensor::<f32, Cpu>::from_data(vec![total], [1]).unwrap()
}
```

## 6.3 每层职责边界 + mod.rs 铁律

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + traits/ + utils/ + oyta_tensor | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! traits 层说明

pub mod loss_fn;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::reduction::Reduction;
pub use traits::loss_fn::LossFn;
pub use utils::log_softmax::log_softmax_stable;
pub use impls::cross_entropy::CrossEntropyLoss;
pub use impls::mse::MSELoss;
pub use impls::multi_domain::MultiDomainLoss;

pub mod prelude {
    pub use crate::{LossFn, Reduction, CrossEntropyLoss, MSELoss, MultiDomainLoss};
    pub use oyta_tensor::prelude::*;
}
```

---

# 七、验收标准（扩充版：含四层分层 + MultiDomainLoss 按 active 过滤）

```
## 基础验收（原有）
[ ] cargo check --package oyta-loss 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1"（无 rust-version）
[ ] CrossEntropyLoss.grad 对 logits=[[1,0,0],[0,1,0]] class=[0,1]
    → grad[0] = softmax - one_hot（正确）
[ ] 每个函数有中文注释

## 追加：MultiDomainLoss 验收
[ ] MultiDomainLoss.compute 的辅助损失只对 active_ids 内的领域计算
[ ] 遍历 expert_outs（只含激活领域输出）+ 二次 active.contains(did) 确认
[ ] 未激活领域的 aux_loss 不参与 total → 它们的 Parameter 接收不到梯度
[ ] main_loss 始终对 fused（所有激活专家融合后）计算 → 不稀疏化
[ ] label_smoothing ∈ [0, 1) 时 CrossEntropyLoss.compute 正确平滑处理

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use
[ ] Reduction 枚举归入 types/ 层
[ ] LossFn trait 归入 traits/ 层
[ ] log_softmax_stable 纯数学工具归入 utils/ 层
[ ] CrossEntropyLoss + MSELoss + MultiDomainLoss 归入 impl/ 层
[ ] 依赖方向单向：types → traits/utils → impls
```

---

# oyta-optim

> 下一份：oyta-optim 只写核心结构（SparseAdam 按领域独立 m/v/t + Optimizer trait 稀疏扩展）
> 不重复贴完整内容（oyta-loss 已经证明 Cargo.toml 和模块结构正确）
> 
> Cargo.toml 关键：oyta-tensor = "0.1.1" + oyta-router = { path = "../oyta-router" }
> 核心实现：
>   Optimizer trait：zero_grad_sparse / step_sparse / clip_grad_norm_sparse
>   SparseAdam：HashMap<u32, DomainState{m: HashMap<String,Vec<f32>>, v: HashMap<String,Vec<f32>>, t: u64}>
>     step_sparse(active, pool) → 遍历 active 里的领域参数，每个领域用自己的 t 做 bias correction
>   Scheduler trait：step(&mut dyn Optimizer) + WarmupCosineAnneal / CosineAnneal / StepLR

---

# oyta-trainer

> Cargo.toml 关键：oyta-tensor + oyta-content + oyta-router + oyta-expert + oyta-fusion + oyta-autograd + oyta-loss + oyta-optim（全部 workspace path）
> 核心：SparseTrainer::train_epoch 主循环
>   for (raw, target) in dataloader.iter():
>     analysis = content_analyzer.analyze(&raw)
>     (route_result, mask_set) = router.route(&analysis)
>     active = force_all ? experts.map(|e| e.domain_id()) : route_result.active_indices
>     optimizer.zero_grad_sparse(&active, &mut param_pool)  // 梯度累积步
>     expert_outs = expert_executor.forward_sparse(&feature_vector, &route_result)  // 框架层 continue 跳过
>     fused = fusion.fuse(&expert_outs, &route_result.normalized_weights)
>     logits = task_head.forward(&fused)
>     loss = multi_domain_loss.compute(&fused, &target, &active, &expert_outs)
>     sparse_backward(loss_node_id, &active, &graph)  // domain_id 过滤整子图跳过
>     optimizer.clip_grad_norm_sparse(&active, &mut param_pool, max_norm)
>     optimizer.step_sparse(&active, &mut param_pool)
>     scheduler.step(&mut optimizer)

---

# oyta-serialize

> Cargo.toml：oyta-tensor + oyta-router + oyta-expert + safetensors = "0.6.0"
> 核心：
>   save_sparse_safetensors(pool, path, active_only: Option<&[u32]>)
>     → 遍历 pool.registered_ids()，按领域过滤，每个 Parameter 读 data.lock() → bytes
>     → 组 safetensors header JSON（key = param.name，dtype = F32，shape = [...]，offsets = [...]）
>     → 写 8 bytes u64 header_len（LE）+ header（8 字节对齐）+ binary data
>   load_sparse_safetensors(path, pool, allow_partial)
>     → 读 header → 解析 JSON → 遍历 pool.domain_params_mut(did) → 按 param.name 匹配 → 覆盖 data
>     → shape 严格校验（不匹配返回 error）

---

# oyta-frame

> Cargo.toml：**全部 10 个 crate**（oyta-tensor 唯一 registry + 其余 9 个 workspace path）
> 唯一文件 src/lib.rs：pub use 全部核心类型 + prelude! 宏
> main.rs 示例：use oyta_frame::prelude::*; → 构造所有模块 → SparseTrainer::train

---

# 验收：全部 10 份开发文档

```
[ ] 全部 Cargo.toml 只写 oyta-tensor = "0.1.1"（registry 版本）
[ ] 全部 Cargo.toml 无 rust-version（用户项目 edition=2024 但没指定）
[ ] 全部底层算子调用 oyta_tensor 真实 API：
    Linear::forward → Tensor::matmul + Tensor::add（不手写循环）
    LayerNorm::forward → Tensor::layernorm
    TransformerExpert → flash_attention_multihead（正确引用 oyta_ops）
    Loss::compute → Vec 操作（数学实现，不是算子调用）
[ ] autograd Conv1dMasked / BatchNorm1dMasked backward：
    算子本身正确实现（Conv1d 的互相关 + dilated 反卷积 / BN 简化公式）
    mask 应用正确（通道清零 / padding 清零 / BN 被 mask 特征梯度清零）
[ ] ExpertExecutor.forward_sparse：框架层 for 循环 continue 跳过未激活专家
[ ] SparseParameterPool / SparseAdam：按领域独立
[ ] 每个函数有中文注释
```
