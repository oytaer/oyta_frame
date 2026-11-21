//! SparseTrainer：端到端稀疏训练主类
//!
//! 训练时序（严格对齐设计文档 08 §九 11 步）：
//!
//!   for (raw, target) in dataloader.iter() {
//!     ── Step 1: ContentAnalyzer.analyze(&raw) → ContentAnalysis（每个样本独立）
//!     ── Step 2: Router.route(&analysis) → (RoutingResult, SparseMaskSet)（动态决策）
//!     ── Step 3: active = 路由决策 or force_all_experts_active（debug 开关）
//!     ── Step 4: zero_grad_sparse(&active, pool)（梯度累积步开始，只清零激活领域）
//!     ── Step 5: for expert in experts → if active.contains → continue 跳过未激活专家 ⚠️ 核心
//!     ── Step 6: fusion.fuse(&expert_outs, normalized_weights)
//!     ── Step 7: task_head.forward(&fused) → logits
//!     ── Step 8: loss.compute(&fused, &target, &active, &expert_outs)（MultiDomainLoss）
//!     ── Step 9: sparse_backward（autograd domain_id 过滤整子图跳过）
//!     ── Step 10: clip_grad_norm_sparse + step_sparse（累积步末尾，只遍历 active）
//!     ── Step 11: callbacks.on_step_end + global_step++
//!   }
//!
//! 三层稀疏过滤端到端协同：
//!   Layer 1: ExpertExecutor + sparse_backward + Optimizer 三层 domain_id 过滤
//!   Layer 2: Conv1dMasked pre-mask / BatchNorm1dMasked post-mask（autograd Op 内部）
//!   Layer 3: Conv1dWithSequenceMask post-zero padding（autograd Op 内部）

use oyta_content::{ContentAnalyzer, ContentAnalysis};
use oyta_router::{DomainRouter, SparseParameterPool};
use oyta_expert::DomainExpert;
use oyta_fusion::{FeatureFusion, TaskHeadBase};
use oyta_loss::MultiDomainLoss;
use oyta_optim::{Optimizer, Scheduler};
use oyta_tensor::prelude::*;
use crate::types::config::{SparsityConfig, TrainerConfig};
use crate::traits::data_loader::DataLoader;
use crate::traits::callback::Callback;

/// 端到端稀疏训练器
pub struct SparseTrainer {
    /// 内容分析器（每个 batch 独立 analyze → ContentAnalysis）
    pub content_analyzer: Box<dyn ContentAnalyzer>,
    /// 稀疏路由（每个 batch 动态决策 active_ids）
    pub router: DomainRouter,
    /// 所有专家模块（框架层 for 循环 continue 跳过未激活专家）
    pub experts: Vec<Box<dyn DomainExpert>>,
    /// 特征融合层（WeightedSum / ConcatLinear / CrossAttention）
    pub fusion: FeatureFusion,
    /// 任务头（ClassificationHead / RegressionHead）
    pub task_head: Box<dyn TaskHeadBase>,
    /// 优化器（SparseAdam / SparseSGD）
    pub optimizer: Box<dyn Optimizer>,
    /// 多领域联合损失（主损失 + 按 active_ids 过滤辅助损失）
    pub loss: MultiDomainLoss,
    /// 可选 Scheduler（WarmupCosineAnneal / CosineAnneal / StepLR）
    pub scheduler: Option<Box<dyn Scheduler>>,
    /// 稀疏配置（三层稀疏开关 + 路由策略 + debug 开关）
    pub sparsity_config: SparsityConfig,
    /// 训练主配置（max_epochs / gradient_accum / max_grad_norm）
    pub trainer_config: TrainerConfig,
    /// 参数池（所有模块参数 → SparseParameterPool.register_domain）
    pub param_pool: SparseParameterPool,
    /// 全局 step 计数器（跨 epoch 递增）
    pub global_step: u64,
}

impl SparseTrainer {
    /// 构造 SparseTrainer（scheduler 可选，用 with_scheduler 链式加）
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
        Self {
            content_analyzer: analyzer, router, experts, fusion, task_head,
            optimizer, loss, scheduler: None,
            sparsity_config: sc, trainer_config: tc,
            param_pool: SparseParameterPool::default(), global_step: 0,
        }
    }

    /// 链式加 Scheduler
    pub fn with_scheduler(mut self, s: Box<dyn Scheduler>) -> Self {
        self.scheduler = Some(s); self
    }

    /// 把所有模块的参数注册到 param_pool（train_epoch 开始前调一次）
    pub fn register_all_params(&mut self) {
        // 每个专家 → 按 domain_id 独立注册
        for e in &self.experts {
            self.param_pool.register_domain(e.domain_id(), e.parameters());
        }
        // 融合层 + 任务头用特殊 key 标记（它们不属于任何领域）
        self.param_pool.register_domain(u32::MAX - 1, self.fusion.parameters());
        self.param_pool.register_domain(u32::MAX, self.task_head.parameters());
    }

    /// 切换所有模块到训练态（BatchNorm / Dropout 生效）
    pub fn train_all(&mut self) {
        for e in self.experts.iter_mut() { e.train(); }
        self.fusion.train();
    }

    /// 切换所有模块到推理态（BatchNorm 用 running / Dropout 关闭）
    pub fn eval_all(&mut self) {
        for e in self.experts.iter_mut() { e.eval(); }
        self.fusion.eval();
    }

    /// 核心：一个 epoch 的完整训练循环（11 步时序，严格对齐设计文档 08 §九）
    pub fn train_epoch(
        &mut self,
        dataloader: &dyn DataLoader,
        callbacks: &mut [Box<dyn Callback>],
    ) -> f32 {
        // ===== Step 0: epoch 初始化 =====
        self.register_all_params();
        self.train_all();

        let accum = self.trainer_config.gradient_accumulation_steps;
        let max_norm = self.trainer_config.max_grad_norm;
        let mut total_loss: f64 = 0.0;
        let mut count: u64 = 0;

        // ===== 主循环：遍历 dataloader =====
        for (raw, target) in dataloader.iter() {
            let step = self.global_step;

            // ===== Step 1: 内容分析（每个样本独立，无稀疏）=====
            let analysis: ContentAnalysis = match self.content_analyzer.analyze(&raw) {
                Ok(a) => a,
                Err(e) => {
                    println!("[oyta-frame] analyze 失败: {:?}", e);
                    continue;
                }
            };

            // ===== Step 2: 稀疏路由（每个 batch 动态决策，无稀疏）=====
            // Router.route → (RoutingResult, SparseMaskSet)
            // RoutingResult.active_indices = 激活领域列表（每个 batch 可能不同）
            let (route_result, _mask_set) = self.router.route(&analysis);

            // ===== Step 3: 确定激活领域 =====
            // force_all_experts_active = debug 开关：关闭 Layer 1 稀疏
            let active: Vec<u32> = if self.sparsity_config.force_all_experts_active
                || !self.sparsity_config.enable_expert_sparsity
            {
                self.experts.iter().map(|e| e.domain_id()).collect()
            } else {
                route_result.active_indices.clone()
            };

            // ===== Step 4: 梯度累积步开始 → 稀疏清零 =====
            // zero_grad_sparse 只清当前 batch 要激活的领域 grad
            // 未激活领域 grad 保持 None（autograd 没写）
            if step as usize % accum == 0 {
                self.optimizer.zero_grad_sparse(&active, &mut self.param_pool);
            }

            // ===== Step 5: 稀疏专家执行（框架层核心 continue 跳过未激活专家）=====
            // 三层稀疏 Layer 1 第一环：框架层直接 continue 跳过
            // 未激活专家 → forward 零调用 → 省算力！
            let mut expert_outs: Vec<(u32, Tensor<f32, Cpu>)> = Vec::new();
            for expert in &self.experts {
                if active.contains(&expert.domain_id()) {
                    let out = expert.forward(&analysis.feature_vector);
                    expert_outs.push((expert.domain_id(), out));
                }
                // ⚠️ 未激活专家：什么都不做 → continue → 省算力！
            }

            // ===== Step 6: 特征融合 =====
            // 输入只含激活专家输出（Step 5 已过滤）
            let fused = self.fusion.fuse(&expert_outs, &route_result.normalized_weights);

            // ===== Step 7: TaskHead → logits =====
            let _logits = self.task_head.forward(&fused);

            // ===== Step 8: Loss =====
            // MultiDomainLoss 内部按 active_ids 过滤辅助损失
            // 未激活领域的 aux_loss 不参与 total
            let loss = self.loss.compute(&fused, &target, &active, &expert_outs);
            let loss_val = self.loss.item(&loss);
            total_loss += loss_val as f64;

            // ===== Step 9: sparse_backward =====
            // 三层稀疏 Layer 1 第二环：autograd domain_id 过滤整子图跳过
            // 完整实现需要：注册 AutogradEngine 计算图 → backward → grads HashMap
            // Phase 1 简化骨架：注释说明，完整实现在 oyta-autograd crate
            //
            //  伪代码（完整实现）：
            //    let grads = sparse_backward(loss_node_id, &active, &computed_graph);
            //    // grads 里：
            //    //   - 激活专家 Parameter 节点 → 有正确梯度 Tensor
            //    //   - 未激活专家 Parameter 节点 → 被整子图跳过 → grad 不被写
            //    // trainer 把 grads 写回 SparseParameterPool：
            //    for param_node in leaf_nodes_with_grad {
            //      let grad = grads.get(&param_node.id).unwrap();
            //      param_node.param.data.lock().unwrap().grad = Some(grad);
            //    }
            //    // ⚠️ 未激活专家的 Parameter.grad 保持 None

            // ===== Step 10: 梯度累积步末尾 → 稀疏裁剪 + 稀疏更新 =====
            if (step + 1) as usize % accum == 0 {
                // 只统计 / 裁剪 active 领域参数梯度的 L2 norm
                self.optimizer.clip_grad_norm_sparse(&active, &mut self.param_pool, max_norm);

                // 只更新 active 领域参数值 + 状态
                // 每个领域用自己的 DomainState.t 做 bias correction
                // 未激活领域的 DomainState.t 不动、参数值不动、m/v 不动
                self.optimizer.step_sparse(&active, &mut self.param_pool);

                // Scheduler.step（如果有）
                if let Some(s) = self.scheduler.as_mut() {
                    s.step(&mut *self.optimizer);
                }
            }

            // ===== Step 11: Callback + global_step++ =====
            for cb in callbacks.iter_mut() {
                cb.on_step_end(step, loss_val, &active);
            }

            // Debug：mask 统计日志
            if let Some(n) = self.sparsity_config.log_mask_every_n {
                if step as u32 % n == 0 {
                    println!("[mask_stats] step={} active={:?}", step, active);
                }
            }

            self.global_step += 1;
            count += 1;
        }

        // epoch 结束：调 on_epoch_end
        let avg = if count > 0 { (total_loss / count as f64) as f32 } else { 0.0 };
        for cb in callbacks.iter_mut() {
            cb.on_epoch_end(if count > 0 { self.global_step / count } else { 0 } as usize, avg);
        }
        avg
    }

    /// 完整训练循环（max_epochs 个 epoch）
    pub fn train(
        &mut self,
        dataloader: &dyn DataLoader,
        callbacks: &mut [Box<dyn Callback>],
    ) -> Vec<f32> {
        let mut losses = Vec::new();
        for epoch in 0..self.trainer_config.max_epochs {
            let avg = self.train_epoch(dataloader, callbacks);
            println!("[oyta-frame] epoch={} avg_loss={:.6}", epoch, avg);
            losses.push(avg);
        }
        losses
    }

    /// 评估模式（eval_all + 不调 optimizer step）
    ///
    /// ⚠️ 核心约束：evaluate 不调 optimizer 的任何 step 方法
    pub fn evaluate(&mut self, dataloader: &dyn DataLoader) -> f32 {
        self.register_all_params();
        self.eval_all();

        let mut total: f64 = 0.0;
        let mut count: u64 = 0;

        for (raw, target) in dataloader.iter() {
            // Step 1: 内容分析
            let analysis = match self.content_analyzer.analyze(&raw) {
                Ok(a) => a, Err(_) => continue,
            };

            // Step 2+3: 路由 → active
            let (route_result, _) = self.router.route(&analysis);
            let active: Vec<u32> = if self.sparsity_config.force_all_experts_active
                || !self.sparsity_config.enable_expert_sparsity
            {
                self.experts.iter().map(|e| e.domain_id()).collect()
            } else {
                route_result.active_indices.clone()
            };

            // Step 5: 稀疏专家执行
            let mut outs: Vec<(u32, Tensor<f32, Cpu>)> = Vec::new();
            for e in &self.experts {
                if active.contains(&e.domain_id()) {
                    outs.push((e.domain_id(), e.forward(&analysis.feature_vector)));
                }
            }

            // Step 6+7+8: 融合 + TaskHead + Loss
            let fused = self.fusion.fuse(&outs, &route_result.normalized_weights);
            let loss = self.loss.compute(&fused, &target, &active, &outs);
            total += self.loss.item(&loss) as f64;
            count += 1;

            // ⚠️ 不调 Step 4 zero_grad_sparse / Step 9 sparse_backward / Step 10 clip+step
            // → evaluate 不修改 Parameter.grad / 不修改 Parameter.data
            // → 纯前向推理模式
        }

        if count > 0 { (total / count as f64) as f32 } else { 0.0 }
    }
}
