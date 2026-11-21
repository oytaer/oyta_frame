//! SparseParameterPool：参数按领域独立存储
//!
//! 稀疏优化核心：zero_grad_sparse / step_sparse / clip_grad_norm_sparse 只遍历 active_ids 内领域
//! 复杂度：O(active_params) << O(all_params)
//!
//! 为什么按领域独立？
//!   未激活专家的参数不参与 backward → grad = None → zero_grad_sparse 清 None 很快
//!   optimizer.step_sparse 跳过未激活领域 → 参数更新只在激活领域执行

use std::collections::HashMap;
use crate::utils::param::Parameter;
use oyta_tensor::{Tensor, Cpu};

/// 按领域独立存储的参数池
pub struct SparseParameterPool {
    /// domain_id → 该领域所有 Parameter（权重 + 偏置，按构造顺序）
    domain_params: HashMap<u32, Vec<Parameter>>,
    /// 注册过的领域 ID 稳定顺序（HashMap 遍历顺序不稳定，训练/序列化用）
    order: Vec<u32>,
    /// 总参数量（所有领域所有参数 numel 之和）
    total: usize,
}

impl SparseParameterPool {
    pub fn new() -> Self {
        Self { domain_params: HashMap::new(), order: Vec::new(), total: 0 }
    }

    /// 注册一个领域的全部参数（框架初始化时调用一次）
    pub fn register_domain(&mut self, did: u32, params: Vec<Parameter>) {
        let n: usize = params.iter().map(|p| p.numel()).sum();
        self.domain_params
            .entry(did)
            .and_modify(|v| v.extend(params.clone()))
            .or_insert_with(|| {
                self.order.push(did);
                params
            });
        self.total += n;
    }

    /// 查询某个领域的参数（只读）
    pub fn domain_params(&self, did: u32) -> &[Parameter] {
        self.domain_params.get(&did).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 查询某个领域的参数（可变，trainer/optimizer 内部用）
    pub fn domain_params_mut(&mut self, did: u32) -> &mut Vec<Parameter> {
        self.domain_params
            .entry(did)
            .or_insert_with(|| {
                self.order.push(did);
                Vec::new()
            })
    }

    /// 返回已注册领域 ID 列表（稳定顺序）
    pub fn registered_ids(&self) -> &[u32] {
        &self.order
    }

    /// 总参数量
    pub fn total_params(&self) -> usize {
        self.total
    }

    // ===== 稀疏操作 =====

    /// 稀疏清零（只清激活领域参数的 grad）
    ///
    /// 未激活领域的参数 grad 保持 None（或被前一轮 sparse_backward 自然跳过）
    /// 这样 optimizer.step_sparse 也会跳过它们
    pub fn zero_grad_sparse(&mut self, active: &[u32]) {
        for &did in active {
            if let Some(ps) = self.domain_params.get_mut(&did) {
                for p in ps {
                    p.zero_grad();
                }
            }
        }
    }

    /// 稀疏 step（闭包 f 由 Optimizer 提供，对每个激活领域参数执行一次更新）
    pub fn step_sparse<F>(&mut self, active: &[u32], mut f: F)
    where
        F: FnMut(&mut Parameter),
    {
        for &did in active {
            if let Some(ps) = self.domain_params.get_mut(&did) {
                for p in ps {
                    f(p);
                }
            }
        }
    }

    /// 计算激活领域的参数量（稀疏训练监控）
    pub fn count_active_params(&self, active: &[u32]) -> usize {
        active
            .iter()
            .filter_map(|did| self.domain_params.get(did))
            .flat_map(|ps| ps.iter())
            .map(|p| p.numel())
            .sum()
    }

    /// 计算激活领域所有参数 grad 的 L2 范数（clip_grad_norm_sparse 用）
    pub fn grad_norm_sparse(&self, active: &[u32]) -> f32 {
        let mut sum_sq = 0.0f32;
        for &did in active {
            if let Some(ps) = self.domain_params.get(&did) {
                for p in ps {
                    if let Some(g) = p.grad() {
                        for &v in g.data_slice() {
                            sum_sq += v * v;
                        }
                    }
                }
            }
        }
        sum_sq.sqrt()
    }

    /// 稀疏梯度裁剪（只裁剪激活领域的参数梯度）
    ///
    /// 超过 max_norm 时等比缩放所有激活参数的 grad
    /// 未激活领域 / 冻结参数不参与裁剪
    /// 返回裁剪前的实际 L2 范数（训练日志可记录）
    pub fn clip_grad_norm_sparse(&mut self, active: &[u32], max_norm: f32) -> f32 {
        // Step 1: 计算当前 L2 范数
        let norm = self.grad_norm_sparse(active);
        // 范数未超限或极小（防除零）→ 直接返回
        if norm <= max_norm || norm < 1e-6 {
            return norm;
        }
        // Step 2: 等比缩放因子
        let scale = max_norm / norm;
        // Step 3: 遍历激活领域，逐参数缩放 grad
        for &did in active {
            if let Some(ps) = self.domain_params.get_mut(&did) {
                for p in ps {
                    // 冻结参数不参与裁剪（requires_grad=false 本来就没 grad）
                    if !p.requires_grad {
                        continue;
                    }
                    if let Some(g) = p.grad_mut() {
                        // oyta_tensor::Tensor 只读 data_slice() → 重建新 Tensor 赋值
                        let mut v = g.data_slice().to_vec();
                        for x in v.iter_mut() {
                            *x *= scale;
                        }
                        let shape = g.shape().dims.clone();
                        *g = Tensor::<f32, Cpu>::from_data(v, shape)
                            .expect("clip_grad_norm 重建 grad 失败");
                    }
                }
            }
        }
        norm
    }
}

impl Default for SparseParameterPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};

    /// 测试 zero_grad_sparse / step_sparse 只操作激活领域
    #[test]
    fn test_sparse_ops_only_touch_active() {
        let mut pool = SparseParameterPool::new();
        // domain 0: 一个 Tensor
        let p0 = Parameter::new(Tensor::<f32, Cpu>::randn([4, 4]), "domain_0.test.weight");
        // domain 1: 一个 Tensor
        let p1 = Parameter::new(Tensor::<f32, Cpu>::randn([8, 8]), "domain_1.test.weight");
        pool.register_domain(0, vec![p0]);
        pool.register_domain(1, vec![p1]);

        // 设置 domain 1 的参数有梯度
        let fake_grad = Tensor::<f32, Cpu>::randn([8, 8]);
        let p1_mut = pool.domain_params_mut(1);
        p1_mut[0].set_grad(fake_grad);

        // zero_grad_sparse 只清 domain 0（注意 domain 0 没梯度也不会 panic）
        pool.zero_grad_sparse(&[0]);
        // domain 1 的 grad 应仍在（因为没激活）
        assert!(pool.domain_params(1)[0].has_grad(), "未激活领域的 grad 应保持不变");

        // step_sparse 只对激活领域调用闭包
        let mut called = Vec::new();
        pool.step_sparse(&[1], |p| called.push(p.name.clone()));
        assert_eq!(called.len(), 1);
        assert!(called[0].contains("domain_1"), "step_sparse 应只对 domain 1 执行");
    }

    #[test]
    fn test_grad_norm_sparse() {
        let mut pool = SparseParameterPool::new();
        let p = Parameter::new(Tensor::<f32, Cpu>::zeros([2]), "domain_0.test.weight");
        pool.register_domain(0, vec![p]);

        let p_mut = pool.domain_params_mut(0);
        p_mut[0].set_grad(Tensor::<f32, Cpu>::from_data(vec![3.0, 4.0], [2]).unwrap());
        // L2 norm = sqrt(3² + 4²) = 5.0
        let norm = pool.grad_norm_sparse(&[0]);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    /// 测试 clip_grad_norm_sparse：范数超限后等比缩放 grad
    #[test]
    fn test_clip_grad_norm_sparse() {
        let mut pool = SparseParameterPool::new();
        let p = Parameter::new(Tensor::<f32, Cpu>::zeros([2]), "domain_0.test.weight");
        pool.register_domain(0, vec![p]);

        let p_mut = pool.domain_params_mut(0);
        // grad = [3, 4] → L2 norm = 5.0
        p_mut[0].set_grad(Tensor::<f32, Cpu>::from_data(vec![3.0, 4.0], [2]).unwrap());

        // clip 到 2.5 → scale = 2.5 / 5.0 = 0.5 → grad 变成 [1.5, 2.0]
        let before = pool.clip_grad_norm_sparse(&[0], 2.5);
        assert!((before - 5.0).abs() < 1e-5, "返回裁剪前的实际范数");

        // 验证 grad 确实被缩放
        let after_norm = pool.grad_norm_sparse(&[0]);
        assert!((after_norm - 2.5).abs() < 1e-5, "裁剪后范数应等于 max_norm");

        let grad_data = pool.domain_params(0)[0].grad().unwrap().data_slice();
        assert!((grad_data[0] - 1.5).abs() < 1e-5);
        assert!((grad_data[1] - 2.0).abs() < 1e-5);

        // 再次 clip 到 3.0（当前范数 2.5 < 3.0）→ 不应缩放
        let before2 = pool.clip_grad_norm_sparse(&[0], 3.0);
        assert!((before2 - 2.5).abs() < 1e-5);
        let grad_data2 = pool.domain_params(0)[0].grad().unwrap().data_slice();
        assert!((grad_data2[0] - 1.5).abs() < 1e-5, "未超限时 grad 不应被缩放");
    }

    /// 测试 clip_grad_norm_sparse 只裁剪激活领域
    #[test]
    fn test_clip_grad_norm_only_active() {
        let mut pool = SparseParameterPool::new();
        let p0 = Parameter::new(Tensor::<f32, Cpu>::zeros([2]), "domain_0.w");
        let p1 = Parameter::new(Tensor::<f32, Cpu>::zeros([2]), "domain_1.w");
        pool.register_domain(0, vec![p0]);
        pool.register_domain(1, vec![p1]);

        // 两个领域都设置 grad = [3, 4]，范数各 5.0
        let g = Tensor::<f32, Cpu>::from_data(vec![3.0, 4.0], [2]).unwrap();
        pool.domain_params_mut(0)[0].set_grad(g.clone());
        pool.domain_params_mut(1)[0].set_grad(g);

        // 只激活 domain 0，clip 到 2.5
        pool.clip_grad_norm_sparse(&[0], 2.5);

        // domain 0 的 grad 应该被裁剪到 [1.5, 2.0]
        let g0 = pool.domain_params(0)[0].grad().unwrap().data_slice();
        assert!((g0[0] - 1.5).abs() < 1e-5);

        // domain 1 未激活，grad 应保持 [3, 4]
        let g1 = pool.domain_params(1)[0].grad().unwrap().data_slice();
        assert!((g1[0] - 3.0).abs() < 1e-5);
    }
}
