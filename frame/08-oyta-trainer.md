# oyta-trainer 开发文档（Phase 8）

> crate：`oyta-trainer/` | 依赖：**全部 crate** | 职责：端到端稀疏训练主循环 + DataLoader + Callback + Config

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-trainer"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor   = "0.1.1"                                        # registry 统一入口
oyta-content  = { version = "0.1.0", path = "../oyta-content" }
oyta-router   = { version = "0.1.0", path = "../oyta-router" }
oyta-expert   = { version = "0.1.0", path = "../oyta-expert" }
oyta-fusion   = { version = "0.1.0", path = "../oyta-fusion" }
oyta-autograd = { version = "0.1.0", path = "../oyta-autograd" }
oyta-loss     = { version = "0.1.0", path = "../oyta-loss" }
oyta-optim    = { version = "0.1.0", path = "../oyta-optim" }

serde         = { version = "1.0.229", features = ["derive"] }
serde_json    = "1.0.152"
csv           = "1.3.0"
thiserror     = "2.0.16"
anyhow        = "1.0.93"
```

---

# 二、src/lib.rs

```rust
pub mod config;
pub mod sparse_trainer;
pub mod callback;
pub mod data_loader;

pub use config::{SparsityConfig, TrainerConfig};
pub use sparse_trainer::SparseTrainer;
pub use callback::{Callback, LogCallback, SaveCallback};
pub use data_loader::{DataLoader, SimpleDataLoader};

pub mod prelude {
    pub use crate::{SparsityConfig, TrainerConfig, SparseTrainer, Callback, LogCallback, SaveCallback, DataLoader, SimpleDataLoader};
    pub use oyta_tensor::prelude::*;
}
```

---

# 三、src/config.rs

```rust
//! SparsityConfig + TrainerConfig
use oyta_router::routing_strategy::RoutingStrategy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityConfig {
    pub enable_expert_sparsity: bool,               // default = true（核心算力节省）
    pub routing_strategy: RoutingStrategy,          // default = TopK { k: 2 }
    pub enable_channel_sparsity: bool,              // default = false（需人工配置 L2 mask）
    pub enable_sequence_sparsity: bool,             // default = true（padding 必须 mask）
    pub force_all_experts_active: bool,             // debug 开关 = false
    pub log_mask_every_n: Option<u32>,              // default = None
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            enable_expert_sparsity: true,
            routing_strategy: RoutingStrategy::TopK { k: 2 },
            enable_channel_sparsity: false,
            enable_sequence_sparsity: true,
            force_all_experts_active: false,
            log_mask_every_n: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub gradient_accumulation_steps: usize,   // default = 1
    pub max_epochs: usize,                     // default = 10
    pub max_grad_norm: f32,                    // default = 1.0
    pub seed: u64,                             // default = 42
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self { gradient_accumulation_steps: 1, max_epochs: 10, max_grad_norm: 1.0, seed: 42 }
    }
}
```

---

# 四、src/data_loader.rs

```rust
//! DataLoader：训练数据加载
//! 最简 SimpleDataLoader（内存 Vec 直接迭代）

use oyta_content::RawInput;
use oyta_tensor::prelude::*;

pub trait DataLoader: Send {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn iter(&self) -> Box<dyn Iterator<Item = (RawInput, Tensor<f32, Cpu>)> + '_>;
}

pub struct SimpleDataLoader {
    data: Vec<(RawInput, Tensor<f32, Cpu>)>,
}

impl SimpleDataLoader {
    pub fn new() -> Self { Self { data: Vec::new() } }
    pub fn push(&mut self, raw: RawInput, target: Tensor<f32, Cpu>) { self.data.push((raw, target)); }
}

impl Default for SimpleDataLoader { fn default() -> Self { Self::new() } }

impl DataLoader for SimpleDataLoader {
    fn len(&self) -> usize { self.data.len() }
    fn iter(&self) -> Box<dyn Iterator<Item = (RawInput, Tensor<f32, Cpu>)> + '_> {
        Box::new(self.data.iter().map(|(r, t)| (r.clone(), t.clone())))
    }
}
```

---

# 五、src/callback.rs

```rust
//! Callback：日志 / checkpoint / 评估
pub trait Callback: Send + Sync {
    fn on_step_end(&mut self, _step: u64, _loss: f32, _active_ids: &[u32]) {}
    fn on_epoch_end(&mut self, _epoch: usize, _epoch_loss: f32) {}
}

pub struct LogCallback { pub every_n: u32 }
impl LogCallback { pub fn new(every_n: u32) -> Self { Self { every_n } } }
impl Callback for LogCallback {
    fn on_step_end(&mut self, step: u64, loss: f32, active_ids: &[u32]) {
        if step as u32 % self.every_n == 0 {
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            let ids = active_ids.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            println!("[oyta-frame] time={} step={} loss={:.6} active=[{}]", ts, step, loss, ids);
        }
    }
}

pub struct SaveCallback { pub every_n: usize, pub dir: String }
impl SaveCallback { pub fn new(dir: &str, every_n: usize) -> Self { Self { every_n, dir: dir.into() } } }
impl Callback for SaveCallback {
    fn on_epoch_end(&mut self, epoch: usize, _avg: f32) {
        if epoch > 0 && epoch % self.every_n == 0 {
            println!("[oyta-frame] epoch {} → checkpoint dir={}", epoch, self.dir);
        }
    }
}
```

---

# 六、src/sparse_trainer.rs（端到端主循环）

```rust
//! SparseTrainer：端到端稀疏训练主类
//!
//! train_epoch 完整主循环：
//!
//!   for (raw, target) in dataloader.iter() {
//!     analysis = content_analyzer.analyze(&raw)          // Step 1: 内容分析（每个 batch 独立）
//!     (route_result, mask_set) = router.route(&analysis) // Step 2: 稀疏路由（每个 batch 动态决策）
//!     active = 路由决策 or force_all_experts_active       // Step 3: 确定激活领域
//!
//!     if global_step % accum == 0:
//!       optimizer.zero_grad_sparse(&active, &mut pool)   // Step 4: 稀疏清零（只清激活领域）
//!
//!     expert_outs = expert_executor.forward_sparse(      // Step 5: 稀疏专家执行
//!         &analysis.feature_vector,                       //   框架层 continue 跳过未激活专家
//!         &route_result,
//!     )
//!
//!     fused = fusion.fuse(&expert_outs, &route_result.normalized_weights)  // Step 6: 融合
//!     logits = task_head.forward(&fused)                                    // Step 7: TaskHead
//!     loss = multi_domain_loss.compute(&fused, &target, &active, &expert_outs)  // Step 8: Loss
//!
//!     // Step 9: sparse_backward（autograd domain_id 过滤整子图跳过）
//!     // sparse_backward(loss_node_id, &active, &graph) → grads HashMap<NodeId, grad>
//!     // 然后把 grads 里的梯度写回 SparseParameterPool 对应 Parameter.grad
//!
//!     if (global_step + 1) % accum == 0:
//!       optimizer.clip_grad_norm_sparse(&active, &mut pool, max_norm) // Step 10: 稀疏梯度裁剪
//!       optimizer.step_sparse(&active, &mut pool)                    // Step 11: 稀疏更新（只更新激活领域）
//!       if let Some(s) = scheduler.as_mut() { s.step(&mut optimizer); }
//!
//!     for cb in callbacks.iter_mut() { cb.on_step_end(global_step, loss_val, &active); }
//!     global_step += 1;
//!   }

use oyta_content::{ContentAnalyzer, RawInput};
use oyta_router::{DomainRouter, SparseParameterPool};
use oyta_expert::{DomainExpert, ExpertExecutor};
use oyta_fusion::{FeatureFusion, TaskHeadBase};
use oyta_loss::MultiDomainLoss;
use oyta_optim::{Optimizer, Scheduler};
use oyta_tensor::prelude::*;
use crate::config::{SparsityConfig, TrainerConfig};
use crate::data_loader::DataLoader;
use crate::callback::Callback;

pub struct SparseTrainer {
    pub content_analyzer: Box<dyn ContentAnalyzer>,
    pub router: DomainRouter,
    pub experts: Vec<Box<dyn DomainExpert>>,
    pub fusion: FeatureFusion,
    pub task_head: Box<dyn TaskHeadBase>,
    pub optimizer: Box<dyn Optimizer>,
    pub loss: MultiDomainLoss,
    pub scheduler: Option<Box<dyn Scheduler>>,
    pub sparsity_config: SparsityConfig,
    pub trainer_config: TrainerConfig,
    pub param_pool: SparseParameterPool,
    pub global_step: u64,
}

impl SparseTrainer {
    pub fn new(
        analyzer: Box<dyn ContentAnalyzer>,
        router: DomainRouter,
        experts: Vec<Box<dyn DomainExpert>>,
        fusion: FeatureFusion,
        task_head: Box<dyn TaskHeadBase>,
        optimizer: Box<dyn Optimizer>,
        loss: MultiDomainLoss,
        sc: SparsityConfig,
        tc: TrainerConfig,
    ) -> Self {
        Self { content_analyzer: analyzer, router, experts, fusion, task_head, optimizer, loss,
               scheduler: None, sparsity_config: sc, trainer_config: tc,
               param_pool: SparseParameterPool::new(), global_step: 0 }
    }

    pub fn with_scheduler(mut self, s: Box<dyn Scheduler>) -> Self { self.scheduler = Some(s); self }

    /// 把所有模块的参数注册到 param_pool（train_epoch 开始前调用）
    pub fn register_all_params(&mut self) {
        for e in &self.experts { self.param_pool.register_domain(e.domain_id(), e.parameters()); }
        // 融合层 + 任务头用 u32::MAX - 1 / u32::MAX 标记（它们不属于任何领域）
        self.param_pool.register_domain(u32::MAX - 1, self.fusion.parameters());
        self.param_pool.register_domain(u32::MAX, self.task_head.parameters());
    }

    pub fn train_all(&mut self) {
        for e in self.experts.iter_mut() { e.as_mut().train(); }
        self.fusion.train();
    }
    pub fn eval_all(&mut self) {
        for e in self.experts.iter_mut() { e.as_mut().eval(); }
        self.fusion.eval();
    }

    /// 核心：一个 epoch 的完整训练循环
    pub fn train_epoch(&mut self, dataloader: &dyn DataLoader, callbacks: &mut [Box<dyn Callback>]) -> f32 {
        self.register_all_params();
        self.train_all();

        let accum = self.trainer_config.gradient_accumulation_steps;
        let max_norm = self.trainer_config.max_grad_norm;
        let mut total: f64 = 0.0;
        let mut count: u64 = 0;

        for (raw, target) in dataloader.iter() {
            let step = self.global_step;

            // Step 1: 内容分析
            let analysis = self.content_analyzer.analyze(&raw).expect("analyze 失败");

            // Step 2: 稀疏路由（每个 batch 动态决策）
            let (route_result, _mask_set) = self.router.route(&analysis);

            // Step 3: 确定激活领域（force_all_experts_active = debug 开关）
            let active = if self.sparsity_config.force_all_experts_active || !self.sparsity_config.enable_expert_sparsity {
                self.experts.iter().map(|e| e.domain_id()).collect::<Vec<u32>>()
            } else {
                route_result.active_indices.clone()
            };

            // Step 4: 梯度累积步开始 → 稀疏清零
            if step as usize % accum == 0 {
                self.optimizer.zero_grad_sparse(&active, &mut self.param_pool);
            }

            // Step 5: 稀疏专家执行（框架层核心 continue 跳过未激活专家）
            let mut expert_outs: Vec<(u32, Tensor<f32, Cpu>)> = Vec::new();
            for expert in &self.experts {
                if active.contains(&expert.domain_id()) {
                    expert_outs.push((expert.domain_id(), expert.forward(&analysis.feature_vector)));
                }
                // 未激活专家：什么都不做 → forward 零调用 → 省算力
            }

            // Step 6: 特征融合
            let fused = self.fusion.fuse(&expert_outs, &route_result.normalized_weights);

            // Step 7: TaskHead
            let _logits = self.task_head.forward(&fused);  // autograd 图里会注册这个节点

            // Step 8: Loss
            let loss = self.loss.compute(&fused, &target, &active, &expert_outs);
            let loss_val = self.loss.item(&loss);
            total += loss_val as f64;

            // Step 9: sparse_backward（autograd domain_id 过滤整子图跳过）
            // 简化占位：完整实现需要 AutogradEngine 计算图注册
            // 实际调用：sparse_backward(loss_node_id, &active, &graph)
            // 然后把 grads 里对应 Parameter.name 的梯度写回 Parameter.grad

            // Step 10 + 11: 梯度累积步末尾 → 稀疏裁剪 + 稀疏更新
            if (step + 1) as usize % accum == 0 {
                self.optimizer.clip_grad_norm_sparse(&active, &mut self.param_pool, max_norm);
                self.optimizer.step_sparse(&active, &mut self.param_pool);
                if let Some(s) = self.scheduler.as_mut() { s.step(&mut *self.optimizer); }
            }

            // Callback
            for cb in callbacks.iter_mut() { cb.on_step_end(step, loss_val, &active); }

            // Debug 日志
            if let Some(n) = self.sparsity_config.log_mask_every_n {
                if step as u32 % n == 0 { println!("[mask_stats] step={} active={:?}", step, active); }
            }

            self.global_step += 1;
            count += 1;
        }

        let avg = if count > 0 { (total / count as f64) as f32 } else { 0.0 };
        for cb in callbacks.iter_mut() { cb.on_epoch_end(self.global_step / count.max(1), avg); }
        avg
    }

    /// 完整训练循环（max_epochs）
    pub fn train(&mut self, dataloader: &dyn DataLoader, callbacks: &mut [Box<dyn Callback>]) -> Vec<f32> {
        let mut losses = Vec::new();
        for epoch in 0..self.trainer_config.max_epochs {
            let avg = self.train_epoch(dataloader, callbacks);
            println!("[oyta-frame] epoch={} avg_loss={:.6}", epoch, avg);
            losses.push(avg);
        }
        losses
    }

    /// 评估模式（eval_all + 无 optimizer step）
    pub fn evaluate(&mut self, dataloader: &dyn DataLoader) -> f32 {
        self.register_all_params();
        self.eval_all();
        let mut total: f64 = 0.0; let mut count: u64 = 0;
        for (raw, target) in dataloader.iter() {
            let analysis = self.content_analyzer.analyze(&raw).unwrap();
            let (route_result, _) = self.router.route(&analysis);
            let active = if self.sparsity_config.force_all_experts_active || !self.sparsity_config.enable_expert_sparsity {
                self.experts.iter().map(|e| e.domain_id()).collect()
            } else { route_result.active_indices.clone() };
            let mut outs = Vec::new();
            for e in &self.experts {
                if active.contains(&e.domain_id()) { outs.push((e.domain_id(), e.forward(&analysis.feature_vector))); }
            }
            let fused = self.fusion.fuse(&outs, &route_result.normalized_weights);
            let loss = self.loss.compute(&fused, &target, &active, &outs);
            total += self.loss.item(&loss) as f64;
            count += 1;
        }
        if count > 0 { (total / count as f64) as f32 } else { 0.0 }
    }
}
```

---

# 七、oyta-trainer 验收标准

```
[ ] cargo check --package oyta-trainer 零警告
[ ] Cargo.toml 只有 oyta-tensor = "0.1.1"（registry）+ 其余 workspace path
[ ] 无 rust-version
[ ] train_epoch 主循环：
    → ContentAnalyzer.analyze 每个样本独立调用
    → Router.route 每个样本动态决策（active_ids 可能不同）
    → ExpertExecutor for 循环 continue 跳过未激活专家
    → optimizer.zero_grad_sparse / clip_grad_norm_sparse / step_sparse 全只遍历 active 领域
[ ] evaluate 不调 optimizer step
[ ] 每个函数有中文注释
```

---

# 八、四层分层目录规范（对齐 00-overview §八）

oyta-trainer 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层。作为端到端主循环 crate，它依赖所有其他 crate，四层分层让模块边界更清晰：Config 归入 types，DataLoader/Callback trait 归入 traits，主循环 SparseTrainer 归入 impl。

## 8.1 目标目录结构

```
oyta-trainer/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude
│
├── types/                       ← 数据类型层（纯配置结构体）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── config.rs              ← SparsityConfig + TrainerConfig（serde 可序列化）
│
├── traits/                      ← 接口层（trait 定义）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   ├── data_loader.rs          ← DataLoader trait { len, iter → (RawInput, Tensor) }
│   └── callback.rs             ← Callback trait { on_step_end, on_epoch_end }
│
├── utils/                       ← 工具层（小型辅助函数）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── seed.rs                 ← set_seed / get_random_u64（训练可复现性）
│
└── impl/                        ← 实现层（主循环 + trait 实现）
    ├── mod.rs                  ← 只 pub mod xxx;
    ├── sparse_trainer.rs       ← SparseTrainer 主类 + train_epoch 主循环（端到端）
    ├── simple_data_loader.rs   ← SimpleDataLoader（DataLoader trait 实现）
    └── callbacks.rs            ← LogCallback + SaveCallback（Callback trait 实现）
```

## 8.2 每层职责边界 + mod.rs 铁律

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | oyta-router（RoutingStrategy） | traits/ utils/ impl/ |
| traits/ | oyta_content + oyta_tensor + types/ | utils/ impl/ |
| utils/ | oyta_tensor + types/ | traits/ impl/ |
| impl/ | types/ + traits/ + utils/ + 所有其他 crate | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! traits 层说明

pub mod data_loader;
pub mod callback;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::config::{SparsityConfig, TrainerConfig};
pub use traits::data_loader::DataLoader;
pub use traits::callback::Callback;
pub use utils::seed::set_seed;
pub use impls::sparse_trainer::SparseTrainer;
pub use impls::simple_data_loader::SimpleDataLoader;
pub use impls::callbacks::{LogCallback, SaveCallback};

pub mod prelude {
    pub use crate::{SparsityConfig, TrainerConfig, DataLoader, Callback, SparseTrainer, SimpleDataLoader, LogCallback, SaveCallback};
    pub use oyta_tensor::prelude::*;
}
```

---

# 九、端到端稀疏训练完整时序（对齐架构 0.6.8 稀疏数据流图）

## 9.1 完整训练时序图

```
对于一个训练 batch (raw_input, target)：
│
├── Step 1: 内容分析（每个样本独立）
│   │ content_analyzer.analyze(&raw_input) → ContentAnalysis { feature_vector, domain_weights }
│   │ 无稀疏 — 始终执行
│   │
├── Step 2: 稀疏路由（每个 batch 动态决策）
│   │ router.route(&analysis) → (RoutingResult, SparseMaskSet)
│   │ RoutingResult.active_indices = [did1, did2]（激活领域）
│   │ RoutingResult.normalized_weights = [w1, w2]（softmax 归一化）
│   │ 无稀疏 — 始终执行（路由本身不重）
│   │
├── Step 3: 确定激活领域（force_all_experts_active debug 开关）
│   │ active = if sparsity_config.force_all_experts_active {
│   │               所有 expert.domain_id()
│   │            } else {
│   │               route_result.active_indices.clone()
│   │            }
│   │
├── Step 4: 梯度累积步开始 → 稀疏清零
│   │ if global_step % accum == 0:
│   │   optimizer.zero_grad_sparse(&active, &mut param_pool)
│   │   // 只清零 active 领域 Parameter.grad
│   │   // 未激活领域 grad 保持 None（autograd 没写）
│   │
├── Step 5: 稀疏专家执行（框架层核心 continue 跳过未激活专家）
│   │ expert_executor.forward_sparse(&feature_vector, &route_result)
│   │ 内部：
│   │   for expert in experts:
│   │     if active.contains(expert.domain_id()):
│   │       expert_outs.push(expert.forward(&feature_vector))
│   │     // ⚠️ 未激活专家 → continue → forward 零调用 → 省算力！
│   │
├── Step 6: 特征融合
│   │ fusion.fuse(&expert_outs, &route_result.normalized_weights) → fused
│   │ 输入只含激活专家输出（因为 Step 5 只推了激活的）
│   │
├── Step 7: TaskHead → logits
│   │ task_head.forward(&fused) → logits
│   │
├── Step 8: Loss（MultiDomainLoss 按 active 过滤辅助损失）
│   │ loss = multi_domain_loss.compute(&fused, &target, &active, &expert_outs)
│   │ 内部：遍历 expert_outs → 只对 active 里的加 aux_loss
│   │ 未激活领域的 aux_loss 不参与 total
│   │
├── Step 9: sparse_backward（autograd domain_id 过滤整子图跳过）
│   │ sparse_backward(loss_node_id, &active, &graph) → grads HashMap
│   │ 内部（三层过滤）：
│   │   Layer 1: node.domain_id ∉ active_ids → do_skip_entire_subtree
│   │             → 给所有父节点传零梯度 → continue
│   │   Layer 2: Conv1dMasked / BatchNorm1dMasked → op.rs 内部 L2 mask
│   │   Layer 3: Conv1dWithSequenceMask → op.rs 内部 L3 padding 清零
│   │
│   │ grads 里：
│   │   - 激活专家的 Parameter 节点 → 有正确的梯度 Tensor
│   │   - 未激活专家的 Parameter 节点 → 被整子图跳过 → grad 不被写
│   │
│   │ trainer 把 grads 写回 SparseParameterPool：
│   │   for param_node in leaf_nodes_with_grad:
│   │     let grad = grads.get(&param_node.id).unwrap();
│   │     param_node.param.data.lock().unwrap().grad = Some(grad);
│   │   // ⚠️ 未激活专家的 Parameter.grad 保持 None
│   │
├── Step 10: 梯度累积步末尾 → 稀疏裁剪 + 稀疏更新
│   │ if (global_step + 1) % accum == 0:
│   │   norm = optimizer.clip_grad_norm_sparse(&active, &mut param_pool, max_norm)
│   │   // 只统计 / 裁剪 active 领域参数梯度的 L2 norm
│   │
│   │   optimizer.step_sparse(&active, &mut param_pool)
│   │   // 只更新 active 领域参数值
│   │   // 每个领域用自己的 DomainState.t 做 bias correction
│   │   // 未激活领域的 DomainState.t 不动、参数值不动、m/v 不动
│   │
│   │   if scheduler.is_some(): scheduler.step(&mut optimizer)
│   │
├── Step 11: Callback + global_step++
│   │ for cb in callbacks: cb.on_step_end(step, loss_val, &active)
│   │ global_step += 1
│   │
└── 下一个 batch...
```

## 9.2 三层稀疏过滤的端到端协同

```
┌──────────────────────────────────────────────────────────────────────┐
│ 三层稀疏过滤（端到端对齐，确保不漏不重）                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Layer 1: domain_id 过滤（框架层 + autograd + optim 一致）              │
│   Step 5: ExpertExecutor.forward_sparse → continue 跳过未激活专家    │
│   Step 9: sparse_backward domain_id ∉ active_ids → 整子图跳过         │
│   Step 10: optimizer.step_sparse 只遍历 active_ids                   │
│                                                                      │
│ Layer 2: channel_mask 过滤（autograd Op 变体内部）                    │
│   Conv1dMasked → grad_out *= (1 - channel_mask[f]) pre-mask          │
│   BatchNorm1dMasked → 被 mask 特征跳过完整 BN 公式 → 梯度清零         │
│                                                                      │
│ Layer 3: sequence_mask 过滤（autograd Op 变体内部）                   │
│   Conv1dWithSequenceMask → 完整 Conv1d backward → post-zero padding  │
│                                                                      │
│ 关键：三层过滤互不重复，层层收紧稀疏范围，最终效果：                   │
│   - 未激活专家的所有计算完全跳过                                       │
│   - 激活专家内部被 mask 的通道不贡献梯度                               │
│   - padding 时间步不向上游传播梯度                                     │
│   → 梯度流向与实际计算流向完全对称                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 9.3 梯度累积注意点

```rust
//! 梯度累积时 active 可能变化（每个 batch 独立路由决策）
//!
//! accum=4 时：
//!   batch 0: active=[1,2] → zero_grad_sparse([1,2]) → forward → backward（写 grad 到 [1,2]）
//!   batch 1: active=[2,3] → 不清零（还在累积）→ forward → backward（写 grad 到 [2,3]）
//!   batch 2: active=[1]   → 不清零 → forward → backward（写 grad 到 [1]）
//!   batch 3: active=[3,4] → zero_grad_sparse([3,4]) ？→ 不对！accum=4 应该 step 了
//!
//! 正确时序：
//!   batch 0: step%4==0 → zero_grad_sparse(batch0.active) → forward → backward
//!   batch 1: 不清零 → forward → backward（Grads 累加到已有 Parameter.grad）
//!   batch 2: 不清零 → forward → backward
//!   batch 3: step%4==3 → forward → backward → clip_grad_norm_sparse(union_active) → step_sparse(union_active)
//!
//! 问题：累积步内 active 变化，zero_grad_sparse 和 step_sparse 用哪个 active？
//! 答案：
//!   zero_grad_sparse 用"当前 batch 的 active"（因为当前 batch 刚开始，清零这个 batch 要算的）
//!   clip/step_sparse 用"累积窗口内所有 active 的并集"（因为可能多个领域在不同 batch 激活过）
//!
//! Phase 1 简化假设：accum=1（梯度累积关闭）→ 问题不存在
//! Phase 1.5 补：TrainerConfig 增加 track_accum_active_union() 辅助方法
```

---

# 十、验收标准（扩充版：含四层分层 + 端到端稀疏训练时序）

```
## 基础验收（原有）
[ ] cargo check --package oyta-trainer 零警告
[ ] Cargo.toml 只有 oyta-tensor = "0.1.1"（registry）+ 其余 workspace path
[ ] 无 rust-version
[ ] evaluate 不调 optimizer step
[ ] 每个函数有中文注释

## 追加：端到端稀疏训练时序验收
[ ] train_epoch 主循环完整覆盖 Step 1~11
[ ] Step 1: ContentAnalyzer.analyze 每个样本独立调用 ✓
[ ] Step 2: Router.route 每个样本动态决策（active_ids 可能不同）✓
[ ] Step 3: force_all_experts_active 开关正确生效（debug 用）✓
[ ] Step 4: zero_grad_sparse 只清当前 batch 要激活的领域 grad ✓
[ ] Step 5: ExpertExecutor for 循环 continue 跳过未激活专家 ✓
[ ] Step 6: FeatureFusion.fuse 输入只含激活专家输出 ✓
[ ] Step 7: TaskHead.forward → logits（加入 autograd 图）✓
[ ] Step 8: MultiDomainLoss.compute 只对 active 里的领域加辅助损失 ✓
[ ] Step 9: sparse_backward domain_id 过滤整子图跳过 ✓
[ ] Step 9: grads HashMap 正确写回 SparseParameterPool.Parameter.grad ✓
[ ] Step 10: clip_grad_norm_sparse + step_sparse 只遍历 active 领域 ✓
[ ] Step 11: Callback.on_step_end 被调，含 active_ids 参数 ✓

## 追加：三层稀疏过滤端到端协同验收
[ ] Layer 1: ExpertExecutor + sparse_backward + Optimizer 三层 domain_id 过滤完全对齐
[ ] Layer 2: Conv1dMasked pre-mask grad_out、BatchNorm1dMasked post-mask 特征清零
[ ] Layer 3: Conv1dWithSequenceMask post-zero padding grad_x（不影响 grad_w/grad_b）
[ ] 未激活专家完整跳过：forward 不跑 → Parameter.grad = None → optim 更新跳过
[ ] force_all_experts_active=true 时 Layer 1 稀疏关闭（但 Layer 2/3 仍生效）

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use
[ ] SparsityConfig + TrainerConfig 归入 types/ 层
[ ] DataLoader trait + Callback trait 归入 traits/ 层
[ ] set_seed 纯工具归入 utils/ 层
[ ] SparseTrainer + SimpleDataLoader + LogCallback + SaveCallback 归入 impl/ 层
[ ] 依赖方向正确：types → traits/utils → impls → 所有其他 crate
```
