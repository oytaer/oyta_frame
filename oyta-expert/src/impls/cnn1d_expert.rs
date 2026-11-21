//! CNN1DExpert：Conv1d → BN1d → ReLU → MaxPool1d → Linear
//!
//! 典型文本/音频处理专家：
//!   Conv1d（提取局部特征）→ BatchNorm1d（稳定训练）→ ReLU（非线性）→ MaxPool1d（降采样）→ Linear（输出）
//!
//! 注意：Conv1d/BN1d/MaxPool1d 带 forward 缓存（backward 必需）
//! 这些模块需要 &mut self（因为要写 cache），forward 实现上要做内部 UnsafeCell/RefCell 包一层
//! 或者整体设计改成 DomainExpert::forward 签名允许 &mut self
//!
//! 简化方案：CNN1DExpert 内部用 Mutex 包 Conv1d/BN/Pool 的 cache，
//! forward 保持 &self（DomainExpert trait 签名不变）

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use std::sync::Mutex;
use crate::traits::domain_expert::DomainExpert;
use crate::utils::conv1d::{Conv1d, Conv1dConfig};
use crate::utils::batchnorm1d::BatchNorm1d;
use crate::utils::pool::MaxPool1d;
use crate::utils::linear::Linear;

/// CNN1d 专家配置
#[derive(Debug, Clone)]
pub struct CNN1DConfig {
    pub in_channels: usize,
    pub conv_out_channels: usize,
    pub conv_kernel_size: usize,
    pub conv_stride: usize,
    pub conv_padding: usize,
    pub pool_kernel_size: usize,
    pub pool_stride: usize,
    pub seq_len: usize,    // 输入序列长度
    pub output_dim: usize,  // 最终输出维度
}

/// Conv1d → BN1d → ReLU → MaxPool1d → Linear
pub struct CNN1DExpert {
    did: u32,
    in_channels: usize,
    out_dim: usize,
    /// 内部带缓存的模块用 Mutex 包（让 forward 保持 &self）
    conv: Mutex<Conv1d>,
    bn:   Mutex<BatchNorm1d>,
    pool: Mutex<MaxPool1d>,
    /// Linear 无缓存问题，直接持有
    proj: Linear,
}

impl CNN1DExpert {
    /// 构造 CNN1D 专家
    pub fn new(did: u32, cfg: CNN1DConfig) -> Self {
        let conv_cfg = Conv1dConfig {
            in_channels: cfg.in_channels,
            out_channels: cfg.conv_out_channels,
            kernel_size: cfg.conv_kernel_size,
            stride: cfg.conv_stride,
            padding: cfg.conv_padding,
        };
        let out_len_after_conv = conv_cfg.out_seq_len(cfg.seq_len);
        let pool_cfg = super::super::utils::pool::PoolConfig {
            kernel_size: cfg.pool_kernel_size,
            stride: cfg.pool_stride,
            padding: 0,
        };
        let out_len_after_pool = pool_cfg.out_seq_len(out_len_after_conv);
        // pool 输出展平 → Linear
        let flat_dim = cfg.conv_out_channels * out_len_after_pool;

        Self {
            did,
            in_channels: cfg.in_channels,
            out_dim: cfg.output_dim,
            conv: Mutex::new(Conv1d::new(did, "cnn.conv", conv_cfg)),
            bn:   Mutex::new(BatchNorm1d::new(did, "cnn.bn", cfg.conv_out_channels)),
            pool: Mutex::new(MaxPool1d::new(cfg.pool_kernel_size, cfg.pool_stride, 0)),
            proj: Linear::new(did, "cnn.proj", flat_dim, cfg.output_dim),
        }
    }
}

impl DomainExpert for CNN1DExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_channels }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { oyta_tensor::is_training_mode() }

    /// forward 链路：Conv1d → BN1d → ReLU → MaxPool1d → flatten → Linear
    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // x: [batch, in_c, seq_len]
        let batch = x.shape().dims[0];

        // Step 1: Conv1d [batch, out_c, out_len_conv]
        let mut conv = self.conv.lock().unwrap();
        let conv_out = conv.forward(x);
        drop(conv);

        // Step 2: Conv 输出展平成 2D 给 BN → 做完 reshape 回来
        let conv_shape = conv_out.shape().clone();
        let out_c = conv_shape.dims[1];
        let out_len = conv_shape.dims[2];
        let flat_for_bn = conv_out.reshape([batch * out_c, out_len])
            .expect("CNN1D reshape conv out 失败");

        // Step 3: BN1d（在 out_len 维度上归一化）
        let mut bn = self.bn.lock().unwrap();
        let bn_out = bn.forward(&flat_for_bn);
        drop(bn);

        // Step 4: ReLU（gelu 也行，这里用 relu 简洁）
        let relu_out = bn_out.reshape([batch, out_c, out_len])
            .expect("CNN1D reshape back 失败");
        let relu_data: Vec<f32> = relu_out.data_slice().iter()
            .map(|&v| if v > 0.0 { v } else { 0.0 })
            .collect();
        let relu_out = Tensor::<f32, Cpu>::from_data(relu_data, [batch, out_c, out_len])
            .expect("CNN1D relu 构造失败");

        // Step 5: MaxPool1d [batch, out_c, out_len_pool]
        let mut pool = self.pool.lock().unwrap();
        let pool_out = pool.forward(&relu_out);
        drop(pool);

        // Step 6: flatten → [batch, flat_dim]
        let pool_shape = pool_out.shape();
        let flat_dim = pool_shape.dims.iter().product::<usize>();
        let flat = pool_out.reshape([batch, flat_dim])
            .expect("CNN1D flatten 失败");

        // Step 7: Linear 输出
        self.proj.forward(&flat)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.conv.lock().unwrap().parameters());
        all.extend(self.bn.lock().unwrap().parameters());
        all.extend(self.proj.parameters());
        all
    }
}
