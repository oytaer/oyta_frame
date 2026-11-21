//! oyta-optim：稀疏训练优化器集 + Scheduler
//!
//! 四层分层目录：
//! types/       DomainState { m, v, t } — 单个领域的 Adam/SGD 独立状态
//! traits/      Optimizer trait（三个稀疏核心方法）+ Scheduler trait
//! utils/       compute_sparse_grad_norm（L2 grad norm 共享）
//! impls/       SparseAdam + SparseSGD + WarmupCosineAnneal/CosineAnneal/StepLR
//!
//! 核心设计：Optimizer 按领域独立维护状态（HashMap<u32, DomainState>）
//!   → 稀疏训练下每个 batch 只激活 K 个领域 → 未激活领域的 m/v/t 不动
//!   → 保证 bias correction (1 - β^{t+1}) 用的是该领域自己的有效步数

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 顶层导出 =====
pub use types::domain_state::DomainState;
pub use traits::optimizer::Optimizer;
pub use traits::scheduler::Scheduler;
pub use impls::sparse_adam::SparseAdam;
pub use impls::sparse_sgd::SparseSGD;
pub use impls::schedulers::{WarmupCosineAnneal, CosineAnneal, StepLR};

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{
        DomainState, Optimizer, Scheduler,
        SparseAdam, SparseSGD,
        WarmupCosineAnneal, CosineAnneal, StepLR,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool};
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};
    use oyta_router::{Parameter, SparseParameterPool, build_param};

    // ===== 辅助：构造带梯度的 Parameter =====
    fn make_param(name: &str, values: &[f32], grad_values: Option<&[f32]>) -> Parameter {
        let shape = [values.len()];
        let mut p = Parameter::new(
            Tensor::<f32, Cpu>::from_data(values.to_vec(), shape).unwrap(),
            name,
        );
        if let Some(gv) = grad_values {
            p.set_grad(Tensor::<f32, Cpu>::from_data(gv.to_vec(), shape).unwrap());
        }
        p
    }

    // =========================================
    // 验收标准 #1：DomainState 基本功能
    // =========================================
    #[test]
    fn test_domain_state_ensure_and_has() {
        let mut ds = DomainState::new();
        assert!(!ds.has_param("w"));
        ds.ensure("w", 3);
        assert!(ds.has_param("w"));
        assert_eq!(ds.m.get("w").unwrap().len(), 3);
        assert_eq!(ds.v.get("w").unwrap().len(), 3);
        assert_eq!(ds.t, 0);
    }

    // =========================================
    // 验收标准 #2：SparseAdam 按领域独立 t
    // =========================================
    #[test]
    fn test_sparse_adam_domain_independent_t() {
        // 构造 3 个领域各一个参数
        let mut pool = SparseParameterPool::default();
        pool.register_domain(1, vec![make_param("domain_1.p1", &[1.0, 2.0], Some(&[0.1, 0.2]))]);
        pool.register_domain(0, vec![make_param("domain_0.p1", &[3.0, 4.0], Some(&[0.3, 0.4]))]);
        pool.register_domain(2, vec![make_param("domain_2.p1", &[5.0, 6.0], None)]); // 未激活

        let mut opt = SparseAdam::new(0.01);

        // 第一次：active=[1,0]（领域 2 不激活）
        opt.step_sparse(&[1, 0], &mut pool);
        assert_eq!(opt.domain_state(1).unwrap().t, 1);
        assert_eq!(opt.domain_state(0).unwrap().t, 1);
        assert!(opt.domain_state(2).is_none() || opt.domain_state(2).unwrap().t == 0);

        // 第二次：active=[1,0]（领域 1,0 再激活，2 还是不动）
        opt.step_sparse(&[1, 0], &mut pool);
        assert_eq!(opt.domain_state(1).unwrap().t, 2);
        assert_eq!(opt.domain_state(0).unwrap().t, 2);

        // 未激活领域 2 的 DomainState 不存在 / t=0
        if let Some(s2) = opt.domain_state(2) { assert_eq!(s2.t, 0); }
    }

    // =========================================
    // 验收标准 #3：SparseAdam 未激活领域参数值不变
    // =========================================
    #[test]
    fn test_sparse_adam_unactivated_domain_unchanged() {
        let mut pool = SparseParameterPool::default();
        pool.register_domain(1, vec![make_param("d1.w", &[1.0], Some(&[0.5]))]);
        pool.register_domain(2, vec![make_param("d2.w", &[5.0], None)]);

        // 先备份领域 2 的参数值
        let d2_before: Vec<f32> = pool.domain_params(2)[0].data.lock().unwrap().data_slice().to_vec();

        let mut opt = SparseAdam::new(0.01);
        opt.step_sparse(&[1], &mut pool); // 只激活领域 1

        // 领域 1 参数变了
        let d1_after: Vec<f32> = pool.domain_params(1)[0].data.lock().unwrap().data_slice().to_vec();
        assert!((d1_after[0] - 1.0).abs() > 1e-10, "激活领域参数应该被更新");

        // 领域 2 参数完全不变
        let d2_after: Vec<f32> = pool.domain_params(2)[0].data.lock().unwrap().data_slice().to_vec();
        assert_eq!(d2_before, d2_after, "未激活领域参数应该完全不变");
    }

    // =========================================
    // 验收标准 #4：clip_grad_norm_sparse 只裁剪激活领域
    // =========================================
    #[test]
    fn test_clip_grad_norm_sparse_only_active() {
        let mut pool = SparseParameterPool::default();
        // 领域 1: grad=[3,4] → norm=5
        pool.register_domain(1, vec![make_param("d1.w", &[0.0, 0.0], Some(&[3.0, 4.0]))]);
        // 领域 2: grad=[3,4]（同样范数）
        pool.register_domain(2, vec![make_param("d2.w", &[0.0, 0.0], Some(&[3.0, 4.0]))]);

        let mut opt = SparseAdam::new(0.01);

        // 只裁剪领域 1（max_norm=2）
        let before = opt.clip_grad_norm_sparse(&[1], &mut pool, 2.0);
        assert!((before - 5.0).abs() < 1e-5); // 裁剪前范数=5

        // 领域 1 的 grad 应该被 scale：3×(2/5)=1.2, 4×(2/5)=1.6
        let d1_grad = pool.domain_params(1)[0].grad().unwrap().data_slice();
        assert!((d1_grad[0] - 1.2).abs() < 1e-5);
        assert!((d1_grad[1] - 1.6).abs() < 1e-5);

        // 领域 2 的 grad 完全不变（没被裁剪）
        let d2_grad = pool.domain_params(2)[0].grad().unwrap().data_slice();
        assert_eq!(d2_grad[0], 3.0);
        assert_eq!(d2_grad[1], 4.0);
    }

    // =========================================
    // 验收标准 #5：WarmupCosineAnneal 关键点验证
    // =========================================
    #[test]
    fn test_warmup_cosine_anneal_key_points() {
        let mut opt = SparseAdam::new(0.0); // lr 不重要，scheduler 会 set_lr
        let mut sch = WarmupCosineAnneal::new(10, 100, 0.1, 1e-5);

        // step 0 → lr = min_lr
        assert!((sch.get_lr() - 1e-5).abs() < 1e-8);

        // step 10（warmup 结束）→ lr ≈ max_lr（允许小误差）
        for _ in 0..10 { sch.step(&mut opt); }
        assert!((sch.get_lr() - 0.1).abs() < 1e-4, "warmup 结束 lr 应接近 max_lr");
        assert_eq!(sch.step_count(), 10);

        // step 100（total）→ lr ≈ min_lr
        for _ in 0..90 { sch.step(&mut opt); }
        assert!(sch.get_lr() < 1e-4, "total 结束 lr 应接近 min_lr");
        assert_eq!(sch.step_count(), 100);
    }

    // =========================================
    // CosineAnneal 关键点验证
    // =========================================
    #[test]
    fn test_cosine_anneal_endpoints() {
        let mut opt = SparseAdam::new(0.0);
        let mut sch = CosineAnneal::new(100, 0.1, 1e-5);

        // step 0 → current = max_lr（还没 step）
        assert!((sch.get_lr() - 0.1).abs() < 1e-8);

        // step 100 → 接近 min_lr
        for _ in 0..100 { sch.step(&mut opt); }
        assert!(sch.get_lr() < 1e-4);
    }

    // =========================================
    // StepLR 衰减验证
    // =========================================
    #[test]
    fn test_step_lr_decay() {
        let mut opt = SparseAdam::new(0.0);
        let mut sch = StepLR::new(0.1, 10, 0.5); // 每 10 步 lr × 0.5

        for _ in 0..9 { sch.step(&mut opt); }
        assert!((sch.get_lr() - 0.1).abs() < 1e-8); // step 1-9 还没衰减

        sch.step(&mut opt);
        assert!((sch.get_lr() - 0.05).abs() < 1e-8); // step 10：0.1 × 0.5 = 0.05

        for _ in 0..10 { sch.step(&mut opt); }
        assert!((sch.get_lr() - 0.025).abs() < 1e-8); // step 20：0.05 × 0.5 = 0.025
    }

    // =========================================
    // zero_grad_sparse 默认实现只清激活领域
    // =========================================
    #[test]
    fn test_zero_grad_sparse_only_active() {
        let mut pool = SparseParameterPool::default();
        pool.register_domain(1, vec![make_param("d1.w", &[0.0], Some(&[1.0]))]);
        pool.register_domain(2, vec![make_param("d2.w", &[0.0], Some(&[1.0]))]);

        let mut opt = SparseAdam::new(0.01);
        opt.zero_grad_sparse(&[1], &mut pool); // 只清领域 1

        assert!(!pool.domain_params(1)[0].has_grad(), "激活领域 grad 应被清零");
        assert!(pool.domain_params(2)[0].has_grad(), "未激活领域 grad 应保留");
    }

    // =========================================
    // count_updated 辅助方法
    // =========================================
    #[test]
    fn test_count_updated() {
        let mut pool = SparseParameterPool::default();
        pool.register_domain(1, vec![
            make_param("d1.w1", &[0.0], Some(&[1.0])),
            make_param("d1.w2", &[0.0], None), // 没 grad
        ]);
        pool.register_domain(2, vec![make_param("d2.w", &[0.0], None)]);

        let opt = SparseAdam::new(0.01);
        assert_eq!(opt.count_updated(&[1], &pool), 1); // 领域 1 只有 1 个有 grad
        assert_eq!(opt.count_updated(&[1, 2], &pool), 1); // 领域 2 没 grad 的参数不计
    }

    // =========================================
    // SparseSGD step_sparse + 领域独立 t 验证
    // =========================================
    #[test]
    fn test_sparse_sgd_basic() {
        let mut pool = SparseParameterPool::default();
        pool.register_domain(1, vec![make_param("d1.w", &[1.0], Some(&[0.5]))]);

        let mut opt = SparseSGD::new(0.1).no_nesterov();
        opt.step_sparse(&[1], &mut pool);

        // 动量 v 初始为 0，第一步 v = 0.9×0 + 0.5 = 0.5
        // w -= 0.1 × 0.5 = 0.05 → w = 0.95
        let w_new = pool.domain_params(1)[0].data.lock().unwrap().data_slice()[0];
        assert!((w_new - 0.95).abs() < 1e-5);
    }
}
