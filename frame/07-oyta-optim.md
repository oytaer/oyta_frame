# oyta-optim 开发文档（Phase 7）

> crate：`oyta-optim/` | 依赖：**`oyta-tensor = "0.1.1"` + `oyta-router`** | 职责：Optimizer trait（稀疏扩展）+ SparseAdam（按领域独立 m/v/t）+ Scheduler

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-optim"
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
pub mod optimizer;
pub mod sparse_adam;
pub mod sparse_sgd;
pub mod scheduler;

pub use optimizer::Optimizer;
pub use sparse_adam::SparseAdam;
pub use sparse_sgd::SparseSGD;
pub use scheduler::{Scheduler, WarmupCosineAnneal, CosineAnneal, StepLR};

pub mod prelude {
    pub use crate::{Optimizer, SparseAdam, SparseSGD, Scheduler, WarmupCosineAnneal, CosineAnneal, StepLR};
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool};
}
```

---

# 三、src/optimizer.rs

```rust
//! Optimizer trait：所有优化器基类（稀疏扩展）
//!
//! 三个稀疏核心方法：
//!   zero_grad_sparse(active, pool)    —— 只清零激活领域参数的 grad
//!   step_sparse(active, pool)         —— 只更新激活领域参数的值
//!   clip_grad_norm_sparse(active, pool, max_norm) —— 只统计/裁剪激活领域参数梯度范数

use oyta_router::{Parameter, SparseParameterPool};

pub trait Optimizer: Send + Sync {
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);

    /// 稀疏清零（默认实现：调 SparseParameterPool.zero_grad_sparse）
    fn zero_grad_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        pool.zero_grad_sparse(active);
    }

    /// 稀疏 step（核心由具体优化器实现）
    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool);

    /// 稀疏梯度裁剪（默认实现：遍历激活领域参数，L2 norm 裁剪）
    fn clip_grad_norm_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool, max_norm: f32) -> f32 {
        let mut total_sq: f64 = 0.0;
        for &did in active {
            for p in pool.domain_params(did) {
                if let Some(g) = p.grad() { for &v in g.data_slice() { total_sq += (v as f64) * (v as f64); }}
            }
        }
        let total = total_sq.sqrt();
        if total > max_norm as f64 && total > 0.0 {
            let scale = (max_norm as f64 / total) as f32;
            for &did in active {
                for p in pool.domain_params_mut(did) {
                    if let Some(g) = p.grad_mut() { for v in g.data_slice_mut() { *v *= scale; }}
                }
            }
        }
        total as f32
    }

    fn count_updated(&self, active: &[u32], pool: &SparseParameterPool) -> usize {
        active.iter().flat_map(|did| pool.domain_params(did).iter()).filter(|p| p.has_grad()).count()
    }
}
```

---

# 四、src/sparse_adam.rs（按领域独立 m/v/t）

```rust
//! SparseAdam：按领域独立维护状态
//!
//! 为什么按领域独立？
//!   稀疏训练下每个 batch 只激活 K << total 个领域
//!   未激活领域的参数没被 backward → m/v/t 不应该动
//!   如果全局共享 t → 每次 step_sparse 都 +1，但只更新了 K/total 的参数
//!   → 一阶/二阶矩的衰减系数在不同领域上的"有效步数"不一致
//!   → 按领域独立 t 保证每个被激活领域的 Adam 状态正确累积
//!
//! 数学公式：标准 Adam（按领域独立 t）
//!   m_{t+1} = β1·m_t + (1-β1)·g
//!   v_{t+1} = β2·v_t + (1-β2)·g²
//!   m̂ = m_{t+1} / (1 - β1^{t+1})
//!   v̂ = v_{t+1} / (1 - β2^{t+1})
//!   param -= lr · m̂ / (√v̂ + ε) + lr · wd · param   （解耦 weight decay）

use std::collections::HashMap;
use oyta_router::{Parameter, SparseParameterPool};
use crate::optimizer::Optimizer;

/// 单个领域的 Adam 状态（每个领域独立一份）
pub struct DomainState {
    pub m: HashMap<String, Vec<f32>>,  // 一阶矩：key = param.name
    pub v: HashMap<String, Vec<f32>>,  // 二阶矩
    pub t: u64,                        // 该领域独立的 step 计数器
}

impl DomainState {
    pub fn new() -> Self { Self { m: HashMap::new(), v: HashMap::new(), t: 0 } }
    pub fn ensure(&mut self, p: &Parameter) {
        let n = p.name.clone();
        let numel = p.numel();
        self.m.entry(n.clone()).or_insert_with(|| vec![0.0f32; numel]);
        self.v.entry(n.clone()).or_insert_with(|| vec![0.0f32; numel]);
    }
}

pub struct SparseAdam {
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    states: HashMap<u32, DomainState>,  // 按领域独立
}

impl SparseAdam {
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.0, states: HashMap::new() }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self { self.beta1 = b1; self.beta2 = b2; self }
    pub fn with_wd(mut self, wd: f32) -> Self { self.wd = wd; self }
}

impl Optimizer for SparseAdam {
    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }

    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        for &did in active {
            let state = self.states.entry(did).or_insert_with(DomainState::new);
            state.t += 1;  // 该领域被激活并 step → 计数器 +1
            let t = state.t;
            let b1 = self.beta1; let b2 = self.beta2; let eps = self.eps;
            let lr = self.lr; let wd = self.wd;
            let b1_p = b1.powi(t as i32); let b2_p = b2.powi(t as i32);

            let params = pool.domain_params_mut(did);
            for p in params {
                // 没有梯度 → 跳过
                let grad = match p.grad() { Some(g) if p.requires_grad => g.clone(), _ => continue };
                state.ensure(p);
                let m = state.m.get_mut(&p.name).unwrap();
                let v = state.v.get_mut(&p.name).unwrap();
                let gd = grad.data_slice();
                let numel = gd.len();

                // ===== Adam 逐元素更新 =====
                {
                    let mut data = p.data.lock().unwrap();
                    let dd = data.data_slice_mut();
                    for i in 0..numel {
                        let g = gd[i];
                        m[i] = b1 * m[i] + (1.0 - b1) * g;
                        let gs = g * g;
                        v[i] = b2 * v[i] + (1.0 - b2) * gs;
                        let mh = m[i] / (1.0 - b1_p);
                        let vh = v[i] / (1.0 - b2_p);
                        let update = lr * mh / (vh.sqrt() + eps);
                        // 解耦 weight decay（AdamW 风格）
                        let wd_update = lr * wd * dd[i];
                        dd[i] -= update + wd_update;
                    }
                }
            }
        }
    }
}
```

---

# 五、src/scheduler.rs

```rust
//! Scheduler：学习率调度
//! WarmupCosineAnneal（推荐默认）/ CosineAnneal / StepLR
//! step() 调 optimizer.set_lr() 更新 lr

use crate::optimizer::Optimizer;

pub trait Scheduler: Send + Sync {
    fn step(&mut self, opt: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
    fn step_count(&self) -> u64;
}

/// Warmup + Cosine Anneal
pub struct WarmupCosineAnneal {
    warmup: u64, total: u64, max_lr: f32, min_lr: f32, step: u64, current: f32,
}
impl WarmupCosineAnneal {
    pub fn new(warmup: u64, total: u64, max_lr: f32, min_lr: f32) -> Self {
        Self { warmup, total, max_lr, min_lr, step: 0, current: min_lr }
    }
}
impl Scheduler for WarmupCosineAnneal {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let lr = if self.step < self.warmup {
            let r = self.step as f32 / self.warmup as f32;
            self.min_lr + (self.max_lr - self.min_lr) * r
        } else {
            let cos = ((self.step - self.warmup) as f32 / (self.total - self.warmup).max(1) as f32 * std::f32::consts::PI).cos();
            let lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (cos + 1.0);
            lr.max(self.min_lr)
        };
        self.current = lr;
        opt.set_lr(lr);
    }
    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}

pub struct CosineAnneal { total: u64, max_lr: f32, min_lr: f32, step: u64, current: f32 }
impl CosineAnneal { pub fn new(total: u64, max_lr: f32, min_lr: f32) -> Self { Self { total, max_lr, min_lr, step: 0, current: max_lr } } }
impl Scheduler for CosineAnneal {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let pi = std::f32::consts::PI;
        let cos = (self.step as f32 / self.total.max(1) as f32 * pi).cos();
        let lr = (self.min_lr + 0.5*(self.max_lr - self.min_lr)*(cos+1.0)).max(self.min_lr);
        self.current = lr; opt.set_lr(lr);
    }
    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}

pub struct StepLR { step_size: u64, gamma: f32, initial: f32, step: u64, current: f32 }
impl StepLR { pub fn new(initial: f32, step_size: u64, gamma: f32) -> Self { Self { step_size, gamma, initial, step: 0, current: initial } } }
impl Scheduler for StepLR {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let decays = self.step / self.step_size;
        self.current = self.initial * self.gamma.powi(decays as i32);
        opt.set_lr(self.current);
    }
    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}
```

---

# 六、oyta-optim 验收标准

```
[ ] cargo check --package oyta-optim 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1" + oyta-router（workspace path）
[ ] 无 rust-version
[ ] SparseAdam.step_sparse([1,0])：
    → DomainState.t 对 1 和 0 各 +1
    → DomainState.t 对 2, 3 等未激活领域保持 0
    → 领域 1, 0 的参数值更新（新旧不同）
    → 领域 2, 3 的参数值完全不变（逐元素相等）
[ ] clip_grad_norm_sparse 只统计 / 裁剪激活领域参数梯度
[ ] WarmupCosineAnneal step=0→min_lr, step=warmup→max_lr, step=total→≈min_lr
[ ] 每个函数有中文注释
```

---

# 七、四层分层目录规范（对齐 00-overview §八）

oyta-optim 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层。核心：Optimizer trait + Scheduler trait 归入 traits 层，具体优化器实现归入 impl 层，SparseParameterPool 依赖 oyta-router。

## 7.1 目标目录结构

```
oyta-optim/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude
│
├── types/                       ← 数据类型层（纯结构体）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── domain_state.rs         ← DomainState { m, v, t } — 单个领域的 Adam 独立状态
│
├── traits/                      ← 接口层（只 trait 定义）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   ├── optimizer.rs            ← Optimizer trait（三个稀疏核心方法：zero_grad_sparse / step_sparse / clip_grad_norm_sparse）
│   └── scheduler.rs            ← Scheduler trait { step(&mut dyn Optimizer), get_lr, step_count }
│
├── utils/                       ← 工具层（纯算法）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── grad_norm.rs            ← L2 grad norm 计算（clip_grad_norm_sparse 复用）
│
└── impl/                        ← 实现层（trait 的具体实现）
    ├── mod.rs                  ← 只 pub mod xxx;
    ├── sparse_adam.rs          ← SparseAdam（按领域独立 HashMap<u32, DomainState>）
    ├── sparse_sgd.rs           ← SparseSGD（Momentum + Weight Decay，同样按领域独立状态）
    └── schedulers.rs           ← WarmupCosineAnneal / CosineAnneal / StepLR（Scheduler trait 实现）
```

## 7.2 DomainState 完整结构（types 层定义）

```rust
//! types/domain_state.rs
//! 每个领域独立维护的优化器状态
//!
//! 为什么按领域独立？
//!   稀疏训练下每个 batch 只激活 K << total 个领域
//!   未激活领域的参数没被 backward → m/v/t 不应该动
//!   如果全局共享 t → 每次 step_sparse 都 +1，但只更新了 K/total 的参数
//!   → 一阶/二阶矩的衰减系数在不同领域上的"有效步数"不一致
//!   → 按领域独立 t 保证每个被激活领域的 Adam 状态正确累积

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct DomainState {
    /// 一阶矩：key = param.name（与 SparseParameterPool.Parameter.name 完全一致）
    pub m: HashMap<String, Vec<f32>>,
    /// 二阶矩：key = param.name
    pub v: HashMap<String, Vec<f32>>,
    /// 该领域独立的 step 计数器（每次 step_sparse 对该领域执行时 +1）
    pub t: u64,
}

impl DomainState {
    pub fn new() -> Self { Self::default() }

    /// 确保指定 Parameter 有足够的 m/v 缓存（首次 step 时调用）
    pub fn ensure(&mut self, name: &str, numel: usize) {
        self.m.entry(name.to_string()).or_insert_with(|| vec![0.0f32; numel]);
        self.v.entry(name.to_string()).or_insert_with(|| vec![0.0f32; numel]);
    }

    pub fn has_param(&self, name: &str) -> bool {
        self.m.contains_key(name)
    }
}
```

## 7.3 每层职责边界 + mod.rs 铁律

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + traits/ + utils/ + oyta_router + oyta_tensor | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! traits 层说明

pub mod optimizer;
pub mod scheduler;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::domain_state::DomainState;
pub use traits::optimizer::Optimizer;
pub use traits::scheduler::Scheduler;
pub use impls::sparse_adam::SparseAdam;
pub use impls::sparse_sgd::SparseSGD;
pub use impls::schedulers::{WarmupCosineAnneal, CosineAnneal, StepLR};

pub mod prelude {
    pub use crate::{DomainState, Optimizer, Scheduler, SparseAdam, SparseSGD, WarmupCosineAnneal, CosineAnneal, StepLR};
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool};
}
```

---

# 八、Optimizer trait 完整稀疏方法签名 + SparseAdam 详细设计

## 8.1 Optimizer trait 完整稀疏方法签名（traits/optimizer.rs）

```rust
//! Optimizer trait：所有稀疏训练优化器基类
//!
//! 三个稀疏核心方法是端到端稀疏训练闭环的最后一环：
//!   Layer 1 稀疏：autograd sparse_backward domain_id 过滤 → 未激活专家 Parameter.grad 不被写
//!   Layer 2 稀疏：optimizer.zero_grad_sparse → 只清零激活领域参数的旧 grad
//!   Layer 3 稀疏：optimizer.step_sparse → 只更新激活领域参数的值 + 状态
//!
//! 如果只做 Layer 1（autograd 过滤），不做 Layer 2+3：
//!   → zero_grad 会清掉所有参数的 grad（包括没 grad 的）→ 浪费但不影响正确性
//!   → step 会尝试更新所有参数 → 未激活参数 grad=None → 跳过 → 但浪费遍历
//!   → 加上 Layer 2+3 → 真正的省算力

use oyta_router::{Parameter, SparseParameterPool};

pub trait Optimizer: Send + Sync {
    // ========== 基础方法（非稀疏）==========

    /// 当前学习率
    fn lr(&self) -> f32;

    /// 设置学习率（Scheduler 调）
    fn set_lr(&mut self, lr: f32);

    // ========== 三个稀疏核心方法 ==========

    /// 稀疏清零梯度：只清零 active_ids 内领域的 Parameter.grad
    ///
    /// # 为什么需要？
    ///   SparseParameterPool 按领域注册参数
    ///   一个 batch 只激活 K 个领域 → 只有这 K 个领域的 Parameter 有新的 grad
    ///   未激活领域的 Parameter.grad 可能还保持着上一个 batch 的值
    ///   → 不清零的话 step 会用上上一批的旧梯度！
    ///   → 必须只清激活领域，未激活领域的 grad 保持 None（它们没被 backward 写）
    ///
    /// # 实现
    ///   默认实现：遍历 active_ids → pool.domain_params_mut(did) → p.grad_mut().take()
    fn zero_grad_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        for &did in active {
            for p in pool.domain_params_mut(did) {
                p.clear_grad();  // Parameter 内部把 grad 置 None
            }
        }
        // ⚠️ 注意：不遍历未激活领域！它们的 grad 不会被清零
        // 这没关系，因为它们的 grad 本来就是 None（autograd 没写）
        // 下一次 step_sparse 也只会遍历 active_ids
    }

    /// 稀疏 step：只更新 active_ids 内领域的 Parameter 值
    ///
    /// # 核心行为
    ///   遍历 active_ids 内的参数 → 如果 p.grad().is_some() 才更新
    ///   更新时使用该领域自己的 DomainState.m/v/t（不是全局共享的）
    ///
    /// # 必须由具体优化器实现（不同优化器公式不同）
    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool);

    /// 稀疏梯度裁剪：只统计 / 裁剪 active_ids 内领域参数的梯度范数
    ///
    /// # 返回
    ///   裁剪前的全局 L2 范数（f32），用于 logging
    ///
    /// # 默认实现（可覆盖）
    ///   先遍历 active_ids 内参数算 total_norm
    ///   如果 total_norm > max_norm → 对所有激活参数的 grad 做 scale = max_norm / total_norm
    fn clip_grad_norm_sparse(
        &mut self,
        active: &[u32],
        pool: &mut SparseParameterPool,
        max_norm: f32,
    ) -> f32 {
        // Phase 0 骨架：具体实现见 §四 sparse_adam.rs
        // 算法：先算 L2 norm → 再 scale grad
        unimplemented!("Phase 0 骨架，sparse_adam.rs 有完整实现")
    }

    // ========== 辅助方法 ==========

    /// 统计本次 step_sparse 实际更新了多少个 Parameter
    fn count_updated(&self, active: &[u32], pool: &SparseParameterPool) -> usize {
        active.iter()
            .flat_map(|did| pool.domain_params(*did).iter())
            .filter(|p| p.has_grad())
            .count()
    }
}
```

## 8.2 SparseAdam 按领域独立 m/v/t 详细设计（impl 层）

```rust
//! SparseAdam 完整实现要点（impl/sparse_adam.rs 指导方向）
//!
//! 数学公式（标准 Adam + 解耦 Weight Decay，按领域独立 t）：
//!   对每个 Parameter p ∈ active_domains:
//!     m_{t+1} = β1 · m_t + (1-β1) · grad      // 一阶矩（指数滑动平均）
//!     v_{t+1} = β2 · v_t + (1-β2) · grad²     // 二阶矩
//!     m̂ = m_{t+1} / (1 - β1^{t+1})            // 偏差校正（关键！不然初始步噪声大）
//!     v̂ = v_{t+1} / (1 - β2^{t+1})
//!     update = lr · m̂ / (√v̂ + ε)
//!     wd_update = lr · wd · param              // 解耦 weight decay（AdamW 风格）
//!     param -= update + wd_update
//!
//! 按领域独立 t 的影响：
//!   DomainState.t += 1 只发生在该领域被激活并 step_sparse 时
//!   → 未激活领域 DomainState.t 保持不变
//!   → 每个领域的 bias correction (1 - β^{t+1}) 基于自己的有效步数
//!   → 如果领域 A 在 batch 1 激活、batch 2 不激活、batch 3 再激活：
//!     领域 A 的 t 在 batch 1 结束后 +1，batch 3 结束后 +2（batch 2 没动）
//!     → bias correction 用的是"该领域累积被激活的次数"，不是"全局 step 数"

use std::collections::HashMap;
use oyta_router::{Parameter, SparseParameterPool};
use crate::traits::optimizer::Optimizer;
use crate::types::domain_state::DomainState;

pub struct SparseAdam {
    lr: f32,
    beta1: f32,     // default 0.9
    beta2: f32,     // default 0.999
    eps: f32,       // default 1e-8
    wd: f32,        // default 0.0（Weight Decay，AdamW 解耦）
    states: HashMap<u32, DomainState>,  // ⚠️ 按领域独立
}

impl SparseAdam {
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.0, states: HashMap::new() }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self { self.beta1 = b1; self.beta2 = b2; self }
    pub fn with_wd(mut self, wd: f32) -> Self { self.wd = wd; self }

    /// 暴露 states 供 debug / checkpoint 用
    pub fn domain_state(&self, did: u32) -> Option<&DomainState> { self.states.get(&did) }
}

impl Optimizer for SparseAdam {
    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }

    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        let lr = self.lr;
        let b1 = self.beta1; let b2 = self.beta2; let eps = self.eps; let wd = self.wd;

        for &did in active {
            // ========== 每个领域独立 t ==========
            let state = self.states.entry(did).or_insert_with(DomainState::new);
            state.t += 1;
            let t = state.t;

            // bias correction 系数（用该领域自己的 t！）
            let b1_pow = b1.powi(t as i32);
            let b2_pow = b2.powi(t as i32);

            let params = pool.domain_params_mut(did);
            for p in params {
                // 没有梯度 → 跳过
                let grad = match p.grad() {
                    Some(g) if p.requires_grad => g.clone(),
                    _ => continue,
                };

                // 确保 m/v 缓存存在
                state.ensure(&p.name, grad.numel());

                // 逐元素 Adam 更新
                {
                    let mut data = p.data.lock().unwrap();
                    let mut m = state.m.get_mut(&p.name).unwrap();
                    let mut v = state.v.get_mut(&p.name).unwrap();
                    let gd = grad.data_slice();
                    let dd = data.data_slice_mut();

                    for i in 0..gd.len() {
                        let g = gd[i];

                        // 一阶矩更新
                        m[i] = b1 * m[i] + (1.0 - b1) * g;

                        // 二阶矩更新
                        let g_sq = g * g;
                        v[i] = b2 * v[i] + (1.0 - b2) * g_sq;

                        // 偏差校正
                        let m_hat = m[i] / (1.0 - b1_pow);
                        let v_hat = v[i] / (1.0 - b2_pow);

                        // AdamW 更新
                        let update = lr * m_hat / (v_hat.sqrt() + eps);
                        let wd_update = lr * wd * dd[i];
                        dd[i] -= update + wd_update;
                    }
                }
            }
        }
        // ⚠️ 函数结束时自动 drop &mut state → 释放 borrow，下次循环 entry 可重入
    }
}
```

---

# 九、验收标准（扩充版：含四层分层 + Optimizer trait + SparseAdam 按领域独立状态）

```
## 基础验收（原有）
[ ] cargo check --package oyta-optim 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1" + oyta-router（workspace path）
[ ] 无 rust-version
[ ] WarmupCosineAnneal step=0→min_lr, step=warmup→max_lr, step=total→≈min_lr
[ ] 每个函数有中文注释

## 追加：Optimizer trait 完整验收
[ ] Optimizer trait 有三个稀疏核心方法：zero_grad_sparse(active, pool) / step_sparse(active, pool) / clip_grad_norm_sparse(active, pool, max_norm)
[ ] zero_grad_sparse 默认实现：遍历 active_ids → pool.domain_params_mut(did) → 清零 grad
    不遍历未激活领域！
[ ] step_sparse 必须由具体优化器实现（不同优化器公式不同）
[ ] clip_grad_norm_sparse 默认实现：先算 active 领域参数的 L2 norm → 如果 > max_norm → scale grad
    返回裁剪前 norm 供 logging
[ ] count_updated 辅助方法：active 内 has_grad() 的 Parameter 数量

## 追加：SparseAdam 按领域独立状态验收
[ ] SparseAdam.states: HashMap<u32, DomainState>（按领域独立，不是全局共享）
[ ] DomainState 完整：m: HashMap<String, Vec<f32>> / v: HashMap<String, Vec<f32>> / t: u64
[ ] DomainState.t 每个领域独立维护：该领域被激活 step 时 +1，未激活时不动
[ ] bias correction 用该领域自己的 t：(1 - β^{t+1})，不是全局 step 数
[ ] step_sparse 完整 Adam 公式：m/v 更新 + bias correction + AdamW 解耦 weight decay
[ ] step_sparse 对未激活领域完全不动：它们的 DomainState.t 不变、参数值不变、m/v 不变
[ ] step_sparse 对 active_ids=[1,0]，跑两次后：
    DomainState.t[1]=2, DomainState.t[0]=2, DomainState.t[2]=0（假设领域 2 没激活过）
    领域 1,0 的参数值有更新（新旧不等）
    领域 2 的参数值完全不变（逐元素相等）
[ ] step_sparse 跳过 grad=None 或 !requires_grad 的 Parameter

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use
[ ] DomainState 归入 types/ 层
[ ] Optimizer trait + Scheduler trait 归入 traits/ 层
[ ] grad_norm 纯工具归入 utils/ 层
[ ] SparseAdam + SparseSGD + schedulers 归入 impl/ 层
[ ] 依赖方向单向：types → traits/utils → impls
```
