//! oyta-loss：损失函数集 + MultiDomainLoss 多领域联合损失
//!
//! 四层分层目录：
//! types/       Reduction 枚举
//! traits/      LossFn trait（compute + grad + item + reduction）
//! utils/       log_softmax_stable / softmax_stable（数值稳定）
//! impls/       CrossEntropyLoss + MSELoss + MultiDomainLoss
//!
//! 核心设计：
//! LossFn trait 同时提供 compute（前向损失）+ grad（解析梯度）
//! → autograd 后端可以直接调 grad() 作为 backward op 实现

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 顶层导出 =====
pub use types::reduction::Reduction;
pub use traits::loss_fn::LossFn;
pub use utils::log_softmax::{log_softmax_stable, softmax_stable};
pub use impls::cross_entropy::CrossEntropyLoss;
pub use impls::mse::MSELoss;
pub use impls::multi_domain::MultiDomainLoss;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{LossFn, Reduction, CrossEntropyLoss, MSELoss, MultiDomainLoss};
    pub use oyta_tensor::prelude::*;
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};

    // ===== Reduction 枚举验证 =====
    #[test]
    fn test_reduction_variants() {
        assert_eq!(Reduction::Mean, Reduction::Mean);
        assert_ne!(Reduction::Mean, Reduction::Sum);
        assert_ne!(Reduction::Mean, Reduction::None);
    }

    // ===== log_softmax 数值稳定验证 =====
    #[test]
    fn test_log_softmax_basic() {
        let ls = log_softmax_stable(&[1.0, 2.0, 3.0]);
        // softmax: [0.09, 0.24, 0.67]，log 后: [-2.41, -1.41, -0.41]
        let sum_exp: f32 = ls.iter().map(|x| x.exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-5, "log_softmax 后 exp 求和应为 1");
    }

    #[test]
    fn test_softmax_stable_extreme() {
        // 极端值（[1000, 1001, 1002]）防溢出
        let sm = softmax_stable(&[1000.0, 1001.0, 1002.0]);
        let s: f32 = sm.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "极端值 softmax 求和应为 1");
        assert!(sm[2] > sm[1] && sm[1] > sm[0], "相对顺序应保留");
    }

    // ===== CrossEntropyLoss 前向 =====
    #[test]
    fn test_cross_entropy_compute_basic() {
        // logits=[[1,0,0],[0,1,0]], target=[0,1]（one-hot 等价）
        let logits = Tensor::<f32, Cpu>::from_data(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]).unwrap();
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0, 1.0], [2]).unwrap();
        let ce = CrossEntropyLoss::new(3, Reduction::Mean);
        let loss = ce.compute(&logits, &target);
        let val = loss.data_slice()[0];
        // 两个样本的 loss 应该相同（logits 对称）
        // softmax([1,0,0]) ≈ [0.58, 0.21, 0.21]，-log 0.58 ≈ 0.54
        assert!(val > 0.0 && val < 5.0, "loss 应在合理范围内");
    }

    // ===== CrossEntropyLoss 解析梯度验证（验收标准 #1）=====
    #[test]
    fn test_cross_entropy_grad_exact() {
        // 验证 grad = softmax - one_hot
        let logits = Tensor::<f32, Cpu>::from_data(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]).unwrap();
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0, 1.0], [2]).unwrap();
        let ce = CrossEntropyLoss::new(3, Reduction::Mean);
        let g = ce.grad(&logits, &target);
        let gd = g.data_slice();

        // 手动计算样本 0 的 softmax
        let sm = softmax_stable(&[1.0, 0.0, 0.0]);
        // one_hot class=0: [1, 0, 0]
        // grad[0] = softmax[0] - 1
        let g0_expected = (sm[0] - 1.0) / 2.0; // Mean 除 batch=2
        assert!((gd[0] - g0_expected).abs() < 1e-5, "class 0 的 grad 应为 softmax[0]-1 除以 batch");
        // grad[1] = softmax[1] - 0
        assert!((gd[1] - sm[1] / 2.0).abs() < 1e-5);
        assert!((gd[2] - sm[2] / 2.0).abs() < 1e-5);
    }

    // ===== CrossEntropyLoss Reduction 三种 =====
    #[test]
    fn test_cross_entropy_reduction_modes() {
        let logits = Tensor::<f32, Cpu>::randn([4, 3]);
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0, 1.0, 2.0, 0.0], [4]).unwrap();

        let ce_mean = CrossEntropyLoss::new(3, Reduction::Mean);
        let ce_sum  = CrossEntropyLoss::new(3, Reduction::Sum);
        let ce_none = CrossEntropyLoss::new(3, Reduction::None);

        let loss_mean = ce_mean.compute(&logits, &target).data_slice()[0];
        let loss_sum  = ce_sum.compute(&logits, &target).data_slice()[0];
        let loss_none = ce_none.compute(&logits, &target);
        let loss_none_d = loss_none.data_slice();

        assert!((loss_mean - loss_sum / 4.0).abs() < 1e-5, "Mean × batch 应 ≈ Sum");
        assert_eq!(loss_none.numel(), 4);
        // Mean ≈ none 平均
        let none_mean: f32 = loss_none_d.iter().sum::<f32>() / 4.0;
        assert!((loss_mean - none_mean).abs() < 1e-5);
    }

    // ===== CrossEntropyLoss label_smoothing =====
    #[test]
    fn test_cross_entropy_label_smoothing() {
        let logits = Tensor::<f32, Cpu>::from_data(vec![10.0, -10.0, -10.0], [1, 3]).unwrap();
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0], [1]).unwrap();
        // ε=0.0 → 硬标签，置信度极高（loss ≈ 0）
        let ce_hard = CrossEntropyLoss::new(3, Reduction::Mean);
        let loss_hard = ce_hard.compute(&logits, &target).data_slice()[0];

        // ε=0.1 → 软标签，loss 应比硬标签略大
        let ce_soft = CrossEntropyLoss::new(3, Reduction::Mean).with_label_smoothing(0.1);
        let loss_soft = ce_soft.compute(&logits, &target).data_slice()[0];

        assert!(loss_soft > loss_hard, "label_smoothing 应让 loss 略大");
        assert!(loss_hard < 0.1, "硬标签高置信度 loss 应接近 0");
    }

    // ===== MSELoss 基础 =====
    #[test]
    fn test_mse_loss_basic() {
        let pred = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let tgt = Tensor::<f32, Cpu>::from_data(vec![2.0, 2.0, 2.0], [3]).unwrap();
        let mse = MSELoss::new(Reduction::Mean);
        let loss = mse.compute(&pred, &tgt).data_slice()[0];
        // (1+0+1)/3 = 0.6667
        assert!((loss - 2.0_f32 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mse_grad_exact() {
        let pred = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let tgt = Tensor::<f32, Cpu>::from_data(vec![2.0, 2.0, 2.0], [3]).unwrap();
        let mse = MSELoss::new(Reduction::Mean);
        let g = mse.grad(&pred, &tgt);
        let gd = g.data_slice();
        // grad = (2/N) × (pred - target) = (2/3) × [-1, 0, 1]
        assert!((gd[0] - 2.0_f32 / 3.0 * (-1.0)).abs() < 1e-5);
        assert!((gd[1]).abs() < 1e-5);
        assert!((gd[2] - 2.0_f32 / 3.0 * 1.0).abs() < 1e-5);
    }

    // ===== MultiDomainLoss 按 active 过滤（核心验收）=====
    #[test]
    fn test_multi_domain_loss_only_active() {
        // 主损失：CrossEntropy
        let main: Box<dyn LossFn> = Box::new(CrossEntropyLoss::new(3, Reduction::Mean));
        // 辅助损失：domain 1 和 domain 2 各一个
        let aux1: Box<dyn LossFn> = Box::new(CrossEntropyLoss::new(3, Reduction::Mean));
        let aux2: Box<dyn LossFn> = Box::new(CrossEntropyLoss::new(3, Reduction::Mean));

        let mut mdl = MultiDomainLoss::new(main).with_aux_weight(1.0);
        mdl.with_auxiliary(1, aux1);
        mdl.with_auxiliary(2, aux2);

        let fused = Tensor::<f32, Cpu>::randn([2, 3]);
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0, 1.0], [2]).unwrap();
        let out1 = Tensor::<f32, Cpu>::randn([2, 3]); // domain 1 输出
        let out2 = Tensor::<f32, Cpu>::randn([2, 3]); // domain 2 输出

        // 场景 A：active=[1,2]，expert_outs 两个都有
        let loss_both = mdl.compute(&fused, &target, &[1, 2], &[(1, out1.clone()), (2, out2.clone())]);
        let val_both = loss_both.data_slice()[0];

        // 场景 B：active=[1]，expert_outs 只含 domain 1
        let loss_one = mdl.compute(&fused, &target, &[1], &[(1, out1.clone())]);
        let val_one = loss_one.data_slice()[0];

        // 场景 C：active=[1]，但 expert_outs 里包含了 domain 2（防御性过滤）
        let loss_filtered = mdl.compute(&fused, &target, &[1], &[(1, out1.clone()), (2, out2.clone())]);
        let val_filtered = loss_filtered.data_slice()[0];

        assert!(val_both > val_one, "两个激活领域的总损失应 > 只有一个");
        assert!((val_filtered - val_one).abs() < 1e-5,
            "防御性过滤：active 不含 domain 2 时 expert_outs 里的 domain 2 应被跳过");
    }

    // ===== LossFn trait item 方法 =====
    #[test]
    fn test_loss_fn_item() {
        let ce = CrossEntropyLoss::new(3, Reduction::Mean);
        let logits = Tensor::<f32, Cpu>::randn([4, 3]);
        let target = Tensor::<f32, Cpu>::from_data(vec![0.0, 1.0, 2.0, 0.0], [4]).unwrap();
        let loss = ce.compute(&logits, &target);
        let v = ce.item(&loss);
        assert!((v - loss.data_slice()[0]).abs() < 1e-6, "item 应返回 loss 的标量值");
    }
}
