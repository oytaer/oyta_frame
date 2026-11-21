//! oyta-expert：Phase 3 框架层 crate
//!
//! 四层分层目录结构：
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ lib.rs           顶层统一 pub mod + pub use + prelude               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ types/           数据类型层（纯结构体，零业务逻辑）                  │
//! │   expert_cache.rs    BN/Conv1d/Pool 的 forward 缓存结构            │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ traits/          接口层（只 trait 定义，零实现）                    │
//! │   domain_expert.rs   DomainExpert trait（所有专家统一接口）         │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ utils/           工具层（参数包装 + 算子实现）                      │
//! │   linear.rs          Linear 全连接层（matmul + add）                │
//! │   norm.rs            LayerNorm / RMSNorm                            │
//! │   batchnorm1d.rs     BatchNorm1d（训练态/推理态 + 缓存）            │
//! │   conv1d.rs          Conv1d 一维卷积（纯 Vec forward + 缓存）       │
//! │   pool.rs            MaxPool1d / AvgPool1d（MaxPool 写 argmax 缓存）│
//! │   dropout.rs         re-export oyta_tensor 训练态控制               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ impls/           实现层（DomainExpert trait 具体实现）              │
//! │   expert_executor.rs  ExpertExecutor（框架层稀疏控制 + continue）   │
//! │   mlp_expert.rs       MLPExpert（Linear → LN → GELU → Linear）      │
//! │   cnn1d_expert.rs    CNN1DExpert（Conv1d → BN → ReLU → Pool → Lin）│
//! │   rnn_expert.rs      RNNExpert（骨架）                               │
//! │   transformer_expert.rs  TransformerExpert（LN → FFN → 残差 → proj）│
//! └─────────────────────────────────────────────────────────────────────┘

// ===== 四层分层：types → traits → utils → impls =====
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 公共导出（从子文件直接 pub use，不经过中间 mod.rs）=====
// types 层
pub use types::expert_cache::{BatchNormCache, Conv1dCache, PoolCache};
// traits 层
pub use traits::domain_expert::DomainExpert;
// utils 层
pub use utils::linear::Linear;
pub use utils::norm::{LayerNorm, RMSNorm};
pub use utils::batchnorm1d::BatchNorm1d;
pub use utils::conv1d::{Conv1d, Conv1dConfig};
pub use utils::pool::{MaxPool1d, AvgPool1d, PoolConfig};
pub use utils::dropout::{set_training_mode, is_training_mode, apply_dropout};
// impls 层
pub use impls::expert_executor::ExpertExecutor;
pub use impls::mlp_expert::MLPExpert;
pub use impls::cnn1d_expert::{CNN1DExpert, CNN1DConfig};
pub use impls::rnn_expert::RNNExpert;
pub use impls::transformer_expert::TransformerExpert;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::*;
    pub use oyta_tensor::prelude::*;
    pub use oyta_router::{Parameter, SparseParameterPool, RoutingResult};
}

// ===== 单元测试（冒烟测试）=====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};
    use oyta_router::RoutingResult;
    use serial_test::serial; // 全局 training_mode 测试互斥执行

    // ===== Linear 层 =====
    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new(0, "test", 8, 4);
        let x = Tensor::<f32, Cpu>::randn([2, 8]);
        let y = linear.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 4], "Linear 输出 shape 应为 [2, 4]");
    }

    #[test]
    fn test_linear_parameters_count() {
        let linear = Linear::new(0, "test", 8, 4);
        let params = linear.parameters();
        assert_eq!(params.len(), 2, "Linear 应有 weight + bias 共 2 个参数");
        let total = params.iter().map(|p| p.numel()).sum::<usize>();
        assert_eq!(total, 8 * 4 + 4, "参数量 = in*out + out");
    }

    // ===== LayerNorm =====
    #[test]
    fn test_layernorm_forward_shape() {
        let ln = LayerNorm::new(0, "test", 16);
        let x = Tensor::<f32, Cpu>::randn([3, 16]);
        let y = ln.forward(&x);
        assert_eq!(y.shape().dims, vec![3, 16], "LN 应保持 shape");
    }

    // ===== RMSNorm =====
    #[test]
    fn test_rmsnorm_forward_shape() {
        let rms = RMSNorm::new(0, "test", 16);
        let x = Tensor::<f32, Cpu>::randn([3, 16]);
        let y = rms.forward(&x);
        assert_eq!(y.shape().dims, vec![3, 16]);
    }

    #[test]
    fn test_rmsnorm_values() {
        let rms = RMSNorm::new(0, "test", 4);
        // 全相同值 → RMSNorm 应输出 ≈ gamma（这里 gamma=1）
        let x = Tensor::<f32, Cpu>::from_data(vec![2.0f32; 4], [4]).unwrap();
        let y = rms.forward(&x);
        let yd = y.data_slice();
        for &v in yd {
            assert!((v - 1.0).abs() < 1e-3, "RMSNorm 全相同值应输出 ≈ 1.0");
        }
    }

    // ===== Conv1d =====
    #[test]
    fn test_conv1d_forward_shape() {
        let cfg = Conv1dConfig {
            in_channels: 2, out_channels: 4,
            kernel_size: 3, stride: 1, padding: 0,
        };
        let mut conv = Conv1d::new(0, "test", cfg);
        let x = Tensor::<f32, Cpu>::randn([1, 2, 8]);
        let y = conv.forward(&x);
        // out_len = (8 + 0 - 3)/1 + 1 = 6
        assert_eq!(y.shape().dims, vec![1, 4, 6]);
        assert!(conv.cache().cached_x.is_some(), "Conv1d forward 应缓存 x");
    }

    // ===== MaxPool1d =====
    #[test]
    fn test_maxpool1d_forward_shape_and_argmax() {
        let mut pool = MaxPool1d::new(2, 2, 0);
        let x = Tensor::<f32, Cpu>::randn([1, 3, 8]);
        let y = pool.forward(&x);
        // out_len = (8 + 0 - 2)/2 + 1 = 4
        assert_eq!(y.shape().dims, vec![1, 3, 4]);
        let cache = pool.cache();
        assert!(cache.argmax.is_some(), "MaxPool 应写入 argmax 缓存");
        assert_eq!(cache.argmax.as_ref().unwrap().len(), 1 * 3 * 4);
    }

    // ===== BatchNorm1d 训练态 + 推理态 =====
    #[serial]
    #[test]
    fn test_batchnorm1d_train_updates_running() {
        let saved_mode = oyta_tensor::is_training_mode();
        oyta_tensor::set_training_mode(true);
        let mut bn = BatchNorm1d::new(0, "test", 4);
        let init_running_mean = bn.running_mean.clone();
        let x = Tensor::<f32, Cpu>::from_data(
            vec![10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0],
            [2, 4],
        ).unwrap();
        let _ = bn.forward(&x);
        assert!(bn.running_mean.iter().any(|&v| v.abs() > 0.5),
            "训练态 BN 应更新 running_mean（当前={:?}，初始={:?}）",
            bn.running_mean, init_running_mean);
        assert!(bn.cache().cached_xhat.is_some());
        assert!(bn.cache().cached_inv_std.is_some());
        oyta_tensor::set_training_mode(saved_mode);
    }

    #[serial]
    #[test]
    fn test_batchnorm1d_eval_no_update() {
        let saved_mode = oyta_tensor::is_training_mode();
        oyta_tensor::set_training_mode(false);
        let mut bn = BatchNorm1d::new(0, "test", 4);
        let init_running_mean = bn.running_mean.clone();
        let x = Tensor::<f32, Cpu>::from_data(
            vec![10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0],
            [2, 4],
        ).unwrap();
        let _ = bn.forward(&x);
        assert_eq!(bn.running_mean, init_running_mean, "推理态 BN 不应更新 running_mean");
        assert!(bn.cache().cached_xhat.is_none(), "推理态不应写 BN 缓存");
        oyta_tensor::set_training_mode(saved_mode);
    }

    // ===== MLPExpert 完整 forward =====
    #[test]
    fn test_mlp_expert_forward() {
        let expert = MLPExpert::new(1, 16, 32, 8);
        let x = Tensor::<f32, Cpu>::randn([2, 16]);
        let y = expert.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 8]);
        assert_eq!(expert.domain_id(), 1);
        assert_eq!(expert.input_dim(), 16);
        assert_eq!(expert.output_dim(), 8);
    }

    // ===== ExpertExecutor 稀疏链路 =====
    #[test]
    fn test_expert_executor_sparse() {
        let e0: Box<dyn DomainExpert> = Box::new(MLPExpert::new(0, 16, 32, 8));
        let e1: Box<dyn DomainExpert> = Box::new(MLPExpert::new(1, 16, 32, 8));
        let e2: Box<dyn DomainExpert> = Box::new(MLPExpert::new(2, 16, 32, 8));

        let executor = ExpertExecutor::new(vec![e0, e1, e2]);
        let input = Tensor::<f32, Cpu>::randn([2, 16]);

        // 只激活 did=0 和 did=2 → did=1 被 continue 跳过
        let result = RoutingResult {
            mask: vec![true, false, true],
            active_indices: vec![0, 2],
            normalized_weights: vec![0.7, 0.3],
        };

        let outputs = executor.forward_sparse(&input, &result);
        assert_eq!(outputs.len(), 2, "应只有 2 个输出（did=1 被跳过）");
        assert_eq!(outputs[0].0, 0);
        assert_eq!(outputs[1].0, 2);
    }

    // ===== CNN1DExpert 完整 forward =====
    #[test]
    fn test_cnn1d_expert_forward() {
        let cfg = CNN1DConfig {
            in_channels: 3, conv_out_channels: 8,
            conv_kernel_size: 3, conv_stride: 1, conv_padding: 0,
            pool_kernel_size: 2, pool_stride: 2,
            seq_len: 8, output_dim: 10,
        };
        let expert = CNN1DExpert::new(5, cfg);
        let x = Tensor::<f32, Cpu>::randn([1, 3, 8]);
        let y = expert.forward(&x);
        assert_eq!(y.shape().dims, vec![1, 10], "CNN1D 输出应为 [1, output_dim]");
        assert_eq!(expert.domain_id(), 5);
        assert_eq!(expert.output_dim(), 10);
    }

    // ===== TransformerExpert forward（5 参数构造，对齐设计文档 §九）=====
    #[test]
    fn test_transformer_expert_forward() {
        let expert = TransformerExpert::new(3, 16, 16, 8, 4);
        let x = Tensor::<f32, Cpu>::randn([2, 16]);
        let y = expert.forward(&x);
        assert_eq!(y.shape().dims, vec![2, 8]);
        assert_eq!(expert.domain_id(), 3);
        assert_eq!(expert.input_dim(), 16);
        assert_eq!(expert.output_dim(), 8);
    }

    // ===== TransformerExpert 参数数量验证（严格对齐 7 组参数）=====
    #[test]
    fn test_transformer_expert_parameters_count() {
        let expert = TransformerExpert::new(3, 16, 16, 8, 4);
        let params = expert.parameters();
        // 7 组参数 = 7 对 (weight, bias) = 14 个 Parameter 结构体
        assert_eq!(params.len(), 14, "TransformerExpert 应有 7 组参数（每组 weight+bias）= 14 个 Parameter");
    }

    // ===== apply_dropout 训练态/推理态行为 =====
    #[serial]
    #[test]
    fn test_apply_dropout_inference_returns_clone() {
        oyta_tensor::set_training_mode(false);
        let x = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let y = apply_dropout(&x, 0.5);
        // 推理态应直接 clone（值完全相同）
        let yd = y.data_slice();
        let xd = x.data_slice();
        assert_eq!(xd, yd, "推理态 apply_dropout 应直接 clone 返回");
    }

    #[serial]
    #[test]
    fn test_apply_dropout_rate_zero() {
        oyta_tensor::set_training_mode(true);
        let x = Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let y = apply_dropout(&x, 0.0);
        // rate=0 应直接 clone
        let yd = y.data_slice();
        let xd = x.data_slice();
        assert_eq!(xd, yd, "rate=0 apply_dropout 应直接 clone 返回");
    }
}
