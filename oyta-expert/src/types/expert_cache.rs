//! BatchNorm / Conv1d / Pool 的 forward 缓存结构
//!
//! backward 时需要 forward 中间结果，这些结构在 forward 末尾写入缓存
//! oyta-autograd crate 消费这些缓存来计算梯度

use oyta_tensor::prelude::*;

/// BatchNorm1d forward 缓存（backward 必需）
#[derive(Debug, Clone, Default)]
pub struct BatchNormCache {
    /// 归一化中间结果 x̂ = (x - μ) / √(σ² + ε)，shape [batch, num_features]
    pub cached_xhat: Option<Tensor<f32, Cpu>>,
    /// 每个特征的 inv_std = 1 / √(σ² + ε)，shape [num_features]
    pub cached_inv_std: Option<Tensor<f32, Cpu>>,
}

/// Conv1d forward 缓存（backward 必需）
#[derive(Debug, Clone, Default)]
pub struct Conv1dCache {
    /// forward 输入 x 的副本（backward 计算 ∂L/∂w 需要原始 x）
    pub cached_x: Option<Tensor<f32, Cpu>>,
}

/// MaxPool1d forward 缓存（backward 必需）
#[derive(Debug, Clone, Default)]
pub struct PoolCache {
    /// argmax 索引，shape [batch, channels, out_len]，值是输入位置索引
    pub argmax: Option<Vec<i32>>,
    /// out_len 记录（backward 重建 shape 用）
    pub out_len: usize,
}
