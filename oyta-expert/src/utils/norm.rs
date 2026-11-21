//! LayerNorm / RMSNorm 归一化层
//!
//! LayerNorm 直接调 oyta_tensor::Tensor::layernorm
//! RMSNorm oyta-ops 有 rmsnorm 高层函数，但也自己写一版纯 Vec 实现作 fallback

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};

/// LayerNorm：(x - mean) / sqrt(var + eps) * gamma + beta
/// 直接调 oyta_tensor::Tensor::layernorm
pub struct LayerNorm {
    /// 特征维度
    num_features: usize,
    /// 数值稳定常数（默认 1e-5）
    eps: f32,
    /// 缩放参数 gamma（初始化为 1）
    gamma: Parameter,
    /// 偏移参数 beta（初始化为 0）
    beta: Parameter,
}

impl LayerNorm {
    /// 构造 LayerNorm（gamma=1, beta=0）
    pub fn new(domain_id: u32, module: &str, num_features: usize) -> Self {
        let g = Tensor::<f32, Cpu>::from_data(vec![1.0f32; num_features], [num_features])
            .expect("LayerNorm gamma 构造失败");
        let b = Tensor::<f32, Cpu>::zeros([num_features]);
        Self {
            num_features,
            eps: 1e-5,
            gamma: build_param(domain_id, module, "gamma", g),
            beta:  build_param(domain_id, module, "beta",  b),
        }
    }

    /// 直接调 oyta_tensor::Tensor::layernorm
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let g = self.gamma.data.lock().unwrap().clone();
        let b = self.beta.data.lock().unwrap().clone();
        x.layernorm(&g, &b, self.eps).expect("LayerNorm::forward 失败")
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    pub fn num_features(&self) -> usize { self.num_features }
}

/// RMSNorm：x / sqrt(mean(x²) + eps) * gamma
/// 只缩放，无中心化，比 LayerNorm 更快（少算 mean）
pub struct RMSNorm {
    /// 特征维度
    num_features: usize,
    /// 数值稳定常数（默认 1e-6）
    eps: f32,
    /// 缩放参数 gamma（初始化为 1）
    gamma: Parameter,
}

impl RMSNorm {
    /// 构造 RMSNorm（gamma=1）
    pub fn new(domain_id: u32, module: &str, num_features: usize) -> Self {
        let g = Tensor::<f32, Cpu>::from_data(vec![1.0f32; num_features], [num_features])
            .expect("RMSNorm gamma 构造失败");
        Self {
            num_features,
            eps: 1e-6,
            gamma: build_param(domain_id, module, "gamma", g),
        }
    }

    /// 纯 Vec 实现：逐行算 RMS → 缩放 × gamma
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let rank = x.rank();
        let inner = x.shape().dims[rank - 1];
        let outer = x.numel() / inner;
        let dd = x.data_slice();
        let g = self.gamma.data.lock().unwrap();
        let gd = g.data_slice();

        let mut out = vec![0.0f32; x.numel()];
        // 逐行算 RMS
        for row in 0..outer {
            let row_start = row * inner;
            // 计算该行的均方根
            let mut sum_sq = 0.0f64;
            for f in 0..inner {
                let v = dd[row_start + f] as f64;
                sum_sq += v * v;
            }
            let rms = (sum_sq / inner as f64 + self.eps as f64).sqrt();
            let inv_rms = 1.0 / rms as f32;
            // 每个元素 × inv_rms × gamma
            for f in 0..inner {
                out[row_start + f] = dd[row_start + f] * inv_rms * gd[f];
            }
        }
        Tensor::<f32, Cpu>::from_data(out, x.shape().clone())
            .expect("RMSNorm::forward 构造失败")
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.gamma.clone()]
    }

    pub fn num_features(&self) -> usize { self.num_features }
}
