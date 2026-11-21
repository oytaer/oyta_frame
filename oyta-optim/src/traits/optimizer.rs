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

use oyta_router::SparseParameterPool;
use oyta_tensor::prelude::*;

/// 稀疏训练优化器 trait（Send + Sync → 可跨线程共享）
pub trait Optimizer: Send + Sync {
    // ========== 基础方法（非稀疏）==========

    /// 当前学习率
    fn lr(&self) -> f32;

    /// 设置学习率（Scheduler 调）
    fn set_lr(&mut self, lr: f32);

    // ========== 三个稀疏核心方法 ==========

    /// 稀疏清零梯度：只清零 active_ids 内领域的 Parameter.grad
    ///
    /// # 默认实现
    /// 遍历 active_ids → pool.domain_params_mut(did) → p.zero_grad()
    /// 不遍历未激活领域！它们的 grad 本来就是 None（autograd 没写）
    fn zero_grad_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        for &did in active {
            for p in pool.domain_params_mut(did) {
                p.zero_grad();
            }
        }
    }

    /// 稀疏 step：只更新 active_ids 内领域的 Parameter 值
    ///
    /// 核心行为：遍历 active_ids 内的参数 → 如果 p.grad().is_some() && p.requires_grad 才更新
    /// 更新时使用该领域自己的 DomainState（不是全局共享的）
    ///
    /// # 必须由具体优化器实现（不同优化器公式不同）
    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool);

    /// 稀疏梯度裁剪：只统计 / 裁剪 active_ids 内领域参数的梯度范数
    ///
    /// # 返回
    /// 裁剪前的全局 L2 范数（f32），用于 logging
    ///
    /// # 默认实现
    /// 1. 先遍历 active_ids 内参数算 total_norm（L2）
    /// 2. 如果 total_norm > max_norm → scale = max_norm / total_norm
    /// 3. 对所有激活参数的 grad 做逐元素 × scale
    ///
    /// ⚠️ oyta_tensor 没有 data_slice_mut() → 修改 grad 用重建模式
    fn clip_grad_norm_sparse(
        &mut self,
        active: &[u32],
        pool: &mut SparseParameterPool,
        max_norm: f32,
    ) -> f32 {
        // Step 1: 算 L2 norm
        let mut total_sq: f64 = 0.0;
        for &did in active {
            for p in pool.domain_params(did) {
                if let Some(g) = p.grad() {
                    for &v in g.data_slice() { total_sq += (v as f64) * (v as f64); }
                }
            }
        }
        let total = total_sq.sqrt() as f32;

        // Step 2: 如果超限 → scale grad（重建模式）
        if total > max_norm && total > 1e-12 {
            let scale = max_norm / total;
            for &did in active {
                for p in pool.domain_params_mut(did) {
                    if let Some(g) = p.grad_mut() {
                        let data = g.data_slice();
                        let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
                        let shape = g.shape().dims.clone();
                        *g = Tensor::<f32, Cpu>::from_data(scaled, shape).unwrap();
                    }
                }
            }
        }
        total
    }

    // ========== 辅助方法 ==========

    /// 统计本次 step_sparse 实际更新了多少个 Parameter（active 内 has_grad 的数量）
    fn count_updated(&self, active: &[u32], pool: &SparseParameterPool) -> usize {
        active.iter()
            .flat_map(|did| pool.domain_params(*did).iter())
            .filter(|p| p.has_grad())
            .count()
    }
}
