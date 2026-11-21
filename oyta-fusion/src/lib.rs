//! oyta-fusion：Phase 4 框架层 crate
//!
//! 四层分层目录结构：
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ lib.rs           顶层统一 pub mod + pub use + prelude               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ types/           数据类型层（纯结构体/枚举，零业务逻辑）              │
//! │   fusion_config.rs    FusionStrategy 枚举（WeightedSum/ConcatLinear/CrossAttention）│
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ traits/          接口层（只 trait，零实现）                        │
//! │   fusion_strategy.rs  FusionStrategyTrait（fuse + fused_dim + parameters）│
//! │   task_head.rs        TaskHeadBase（forward + parameters + output_dim）│
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ utils/           工具层（被 impl 层依赖）                           │
//! │   tensor_concat.rs    concat_tensors 任意 dim 拼接（纯 Vec 实现）   │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ impls/           实现层（trait 具体实现）                           │
//! │   feature_fusion.rs     FeatureFusion（策略分发主类）               │
//! │   weighted_sum.rs       WeightedSumFusion（加权求和，无可训练参数）  │
//! │   concat_linear.rs      ConcatLinearFusion（concat → Linear 投影）  │
//! │   cross_attention.rs    CrossAttentionFusion（FlashAttention + out_proj）│
//! │   task_heads.rs         ClassificationHead + RegressionHead        │
//! └─────────────────────────────────────────────────────────────────────┘

// ===== 四层分层：types → traits → utils → impls =====
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 公共导出 =====
// types 层
pub use types::fusion_config::FusionStrategy;
// traits 层
pub use traits::fusion_strategy::FusionStrategyTrait;
pub use traits::task_head::TaskHeadBase;
// utils 层
pub use utils::tensor_concat::concat_tensors;
// impls 层
pub use impls::feature_fusion::FeatureFusion;
pub use impls::weighted_sum::WeightedSumFusion;
pub use impls::concat_linear::ConcatLinearFusion;
pub use impls::cross_attention::CrossAttentionFusion;
pub use impls::task_heads::{ClassificationHead, RegressionHead, TaskHead};

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::*;
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::Parameter;
    pub use oyta_expert::Linear;
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};
    use oyta_expert::Linear;

    // ===== concat_tensors =====
    #[test]
    fn test_concat_2d_dim1() {
        let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        let c = concat_tensors(&[&a, &b], 1);
        assert_eq!(c.shape().dims, vec![2, 4]);
        let cd = c.data_slice();
        assert_eq!(cd, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_2d_dim0() {
        let a = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0], [1, 2]).unwrap();
        let b = Tensor::<f32, Cpu>::from_data(vec![3.0, 4.0], [1, 2]).unwrap();
        let c = concat_tensors(&[&a, &b], 0);
        assert_eq!(c.shape().dims, vec![2, 2]);
        let cd = c.data_slice();
        assert_eq!(cd, &[1.0, 2.0, 3.0, 4.0]);
    }

    // ===== WeightedSum =====
    #[test]
    fn test_weighted_sum_fuse() {
        let e1 = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0], [1, 2]).unwrap();
        let e2 = Tensor::<f32, Cpu>::from_data(vec![3.0, 4.0], [1, 2]).unwrap();
        let ws = WeightedSumFusion::new(2);
        let out = ws.fuse(&[(0, e1), (1, e2)], &[0.3, 0.7]);
        assert_eq!(out.shape().dims, vec![1, 2]);
        let od = out.data_slice();
        // 0.3*1 + 0.7*3 = 2.4, 0.3*2 + 0.7*4 = 3.4
        assert!((od[0] - 2.4).abs() < 1e-5);
        assert!((od[1] - 3.4).abs() < 1e-5);
        // WeightedSum 无可训练参数
        assert!(ws.parameters().is_empty());
    }

    // ===== ConcatLinear =====
    #[test]
    fn test_concat_linear_fuse() {
        let e1 = Tensor::<f32, Cpu>::randn([2, 4]);
        let e2 = Tensor::<f32, Cpu>::randn([2, 4]);
        let cl = ConcatLinearFusion::new(0, 4, 8);
        let out = cl.fuse(&[(0, e1), (1, e2)], &[0.5, 0.5]);
        assert_eq!(out.shape().dims, vec![2, 8]);
        // 有可训练参数（Linear.weight + bias）
        assert_eq!(cl.parameters().len(), 2);
    }

    // ===== FeatureFusion 策略分发 =====
    #[test]
    fn test_feature_fusion_weighted_sum() {
        let fs = FeatureFusion::new(FusionStrategy::WeightedSum, 0, 4);
        assert_eq!(fs.fused_dim(), 4);
        assert!(fs.parameters().is_empty());
    }

    #[test]
    fn test_feature_fusion_concat_linear() {
        let fs = FeatureFusion::new(FusionStrategy::ConcatLinear { proj_dim: 8 }, 0, 4);
        assert_eq!(fs.fused_dim(), 8);
        assert!(!fs.parameters().is_empty());
    }

    #[test]
    fn test_feature_fusion_cross_attention() {
        let fs = FeatureFusion::new(
            FusionStrategy::CrossAttention { num_heads: 4, dropout: 0.1, proj_dim: 8 },
            0, 4,
        );
        assert_eq!(fs.fused_dim(), 8);
        assert!(!fs.parameters().is_empty());
    }

    // ===== Task Heads =====
    #[test]
    fn test_classification_head() {
        let head = ClassificationHead::new(0, 8, 3);
        let x = Tensor::<f32, Cpu>::randn([2, 8]);
        let y = head.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 3]);
        assert_eq!(head.output_dim(), 3);
        assert_eq!(head.parameters().len(), 2);
    }

    #[test]
    fn test_regression_head() {
        let head = RegressionHead::new(0, 8);
        let x = Tensor::<f32, Cpu>::randn([2, 8]);
        let y = head.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 1]);
        assert_eq!(head.output_dim(), 1);
    }

    // ===== FeatureFusion train/eval =====
    #[test]
    fn test_feature_fusion_train_eval() {
        let mut fs = FeatureFusion::new(FusionStrategy::WeightedSum, 0, 4);
        assert!(fs.is_training(), "默认应是训练态");
        fs.eval();
        assert!(!fs.is_training(), "eval() 后应为推理态");
        fs.train();
        assert!(fs.is_training(), "train() 后应为训练态");
    }

    // ===== TaskHead 枚举分发 =====
    #[test]
    fn test_task_head_classification_dispatch() {
        let head = TaskHead::Classification(ClassificationHead::new(0, 8, 3));
        let x = Tensor::<f32, Cpu>::randn([2, 8]);
        let y = head.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 3]);
        assert_eq!(head.output_dim(), 3);
        assert_eq!(head.parameters().len(), 2);
    }

    #[test]
    fn test_task_head_regression_dispatch() {
        let head = TaskHead::Regression(RegressionHead::new(0, 8));
        let x = Tensor::<f32, Cpu>::randn([2, 8]);
        let y = head.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 1]);
        assert_eq!(head.output_dim(), 1);
        assert_eq!(head.parameters().len(), 2);
    }

    // ===== CrossAttention gating 权重 bias 注入 =====
    #[test]
    fn test_cross_attention_gating_bias() {
        // 非均匀权重 → gating bias 应生效（非零输出）
        let e1 = Tensor::<f32, Cpu>::randn([2, 4]);
        let e2 = Tensor::<f32, Cpu>::randn([2, 4]);
        let ca = CrossAttentionFusion::new(0, 4, 2, 0.0, 8);
        // 权重严重倾斜 → gating bias 应放大专家 1 的贡献
        let out = ca.fuse(&[(0, e1), (1, e2)], &[0.1, 0.9]);
        assert_eq!(out.shape().dims, vec![2, 8]);
        // 所有权重 = 1.0 → 无 bias 注入路径
        let e3 = Tensor::<f32, Cpu>::randn([2, 4]);
        let e4 = Tensor::<f32, Cpu>::randn([2, 4]);
        let out_uniform = ca.fuse(&[(0, e3), (1, e4)], &[1.0, 1.0]);
        assert_eq!(out_uniform.shape().dims, vec![2, 8]);
    }

    // ===== CrossAttention =====
    #[test]
    fn test_cross_attention_fuse() {
        let e1 = Tensor::<f32, Cpu>::randn([2, 4]);
        let e2 = Tensor::<f32, Cpu>::randn([2, 4]);
        let ca = CrossAttentionFusion::new(0, 4, 2, 0.0, 8);
        let out = ca.fuse(&[(0, e1), (1, e2)], &[0.5, 0.5]);
        assert_eq!(out.shape().dims, vec![2, 8]);
        // 三组参数 = 6 个 Parameter
        assert_eq!(ca.parameters().len(), 6);
    }
}
