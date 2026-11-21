//! grad_norm：L2 梯度范数工具（Optimizer.clip_grad_norm_sparse 共享）

use oyta_router::SparseParameterPool;

/// 计算激活领域所有参数的 L2 梯度范数
///
/// 返回 sqrt(Σ g[i]²)
pub fn compute_sparse_grad_norm(active: &[u32], pool: &SparseParameterPool) -> f32 {
    let mut total_sq: f64 = 0.0;
    for &did in active {
        for p in pool.domain_params(did) {
            if let Some(g) = p.grad() {
                for &v in g.data_slice() { total_sq += (v as f64) * (v as f64); }
            }
        }
    }
    total_sq.sqrt() as f32
}
