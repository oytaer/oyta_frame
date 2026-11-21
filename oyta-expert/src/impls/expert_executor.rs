//! ExpertExecutor：框架层稀疏控制流
//!
//! 核心设计：
//!   稀疏发生在 for 循环里 continue 跳过未激活专家
//!   未激活专家 forward 零调用、不分配中间 Tensor
//!   参数池自动从所有专家注册（框架初始化时一次）

use oyta_tensor::prelude::*;
use oyta_router::{RoutingResult, SparseParameterPool};
use crate::traits::domain_expert::DomainExpert;

/// 专家执行器（持有所有 DomainExpert + 统一参数池）
pub struct ExpertExecutor {
    /// 所有已注册的领域专家
    pub experts: Vec<Box<dyn DomainExpert>>,
    /// 全局参数池（所有专家参数按领域独立存储）
    pub param_pool: SparseParameterPool,
}

impl ExpertExecutor {
    /// 构造：自动将所有专家的参数注册到参数池
    pub fn new(experts: Vec<Box<dyn DomainExpert>>) -> Self {
        let mut pool = SparseParameterPool::new();
        for e in &experts {
            e.register(&mut pool);
        }
        Self { experts, param_pool: pool }
    }

    /// 稀疏 forward：只执行 RoutingResult.active_indices 内的专家
    ///
    /// 框架层核心稀疏点：
    /// ```ignore
    /// for expert in &self.experts {
    ///     if !result.active_indices.contains(&expert.domain_id()) {
    ///         continue;  // ===== 未激活专家零执行 =====
    ///     }
    ///     outputs.push((did, expert.forward(input)));
    /// }
    /// ```
    pub fn forward_sparse(
        &self,
        input: &Tensor<f32, Cpu>,
        result: &RoutingResult,
    ) -> Vec<(u32, Tensor<f32, Cpu>)> {
        let mut outputs: Vec<(u32, Tensor<f32, Cpu>)> =
            Vec::with_capacity(result.active_indices.len());
        for expert in &self.experts {
            let did = expert.domain_id();
            // ===== 框架层核心稀疏点：未激活专家零执行 =====
            if !result.active_indices.contains(&did) {
                continue;
            }
            outputs.push((did, expert.forward(input)));
        }
        outputs
    }

    /// 所有专家切训练态（Dropout 启用 + BN running 统计更新）
    pub fn train_all(&mut self) {
        for e in self.experts.iter_mut() {
            e.as_mut().train();
        }
    }

    /// 所有专家切推理态（Dropout 禁用 + BN 只读 running 统计）
    pub fn eval_all(&mut self) {
        for e in self.experts.iter_mut() {
            e.as_mut().eval();
        }
    }

    /// 激活专家的数量（监控用）
    pub fn count_active(&self, active_indices: &[u32]) -> usize {
        self.experts.iter()
            .filter(|e| active_indices.contains(&e.domain_id()))
            .count()
    }

    /// 激活专家的参数总数（监控用）
    pub fn count_active_params(&self, active_indices: &[u32]) -> usize {
        self.param_pool.count_active_params(active_indices)
    }
}
