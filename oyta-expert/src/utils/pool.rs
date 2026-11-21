//! MaxPool1d / AvgPool1d 一维池化
//!
//! MaxPool forward 必须同时写入 argmax 索引数组（backward 必需）
//! AvgPool 无缓存需求（backward 稠密均匀分配梯度）

use oyta_tensor::prelude::*;
use crate::types::expert_cache::PoolCache;

/// Pool 配置参数
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl PoolConfig {
    /// 计算输出序列长度
    pub fn out_seq_len(&self, seq_len: usize) -> usize {
        (seq_len + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

/// MaxPool1d 层
pub struct MaxPool1d {
    pub config: PoolConfig,
    /// forward 缓存（argmax 索引）
    cache: PoolCache,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            config: PoolConfig { kernel_size, stride, padding },
            cache: PoolCache::default(),
        }
    }

    /// 纯 Vec 实现 MaxPool1d forward，同时写入 argmax
    ///
    /// x: [batch, channels, seq_len] → out: [batch, channels, out_len]
    pub fn forward(&mut self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let [batch, channels, seq_len] = x.shape().dims[..] else { panic!("MaxPool1d 需要 3D 输入") };
        let k = self.config.kernel_size;
        let stride = self.config.stride;
        let padding = self.config.padding;
        let out_len = self.config.out_seq_len(seq_len);
        let xd = x.data_slice();

        let mut out = vec![0.0f32; batch * channels * out_len];
        let mut argmax = vec![-1i32; batch * channels * out_len];

        for n in 0..batch {
            for c in 0..channels {
                let base = (n * channels + c) * out_len;
                let x_base = (n * channels + c) * seq_len;
                for o in 0..out_len {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = -1i32;
                    for ki in 0..k {
                        let ki_pos = o * stride + ki;
                        // 防 usize 下溢
                        if ki_pos >= padding {
                            let t = ki_pos - padding;
                            if t < seq_len {
                                let v = xd[x_base + t];
                                if v > max_val {
                                    max_val = v;
                                    max_idx = t as i32;
                                }
                            }
                        }
                    }
                    out[base + o] = max_val;
                    argmax[base + o] = max_idx;
                }
            }
        }

        // 写入缓存
        self.cache.argmax = Some(argmax);
        self.cache.out_len = out_len;

        Tensor::<f32, Cpu>::from_data(out, [batch, channels, out_len])
            .expect("MaxPool1d forward 输出构造失败")
    }

    pub fn cache(&self) -> &PoolCache { &self.cache }
    pub fn clear_cache(&mut self) { self.cache = PoolCache::default(); }
}

/// AvgPool1d 层
pub struct AvgPool1d {
    pub config: PoolConfig,
}

impl AvgPool1d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { config: PoolConfig { kernel_size, stride, padding } }
    }

    /// 纯 Vec 实现 AvgPool1d forward
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let [batch, channels, seq_len] = x.shape().dims[..] else { panic!("AvgPool1d 需要 3D 输入") };
        let k = self.config.kernel_size;
        let stride = self.config.stride;
        let padding = self.config.padding;
        let out_len = self.config.out_seq_len(seq_len);
        let xd = x.data_slice();

        let mut out = vec![0.0f32; batch * channels * out_len];
        for n in 0..batch {
            for c in 0..channels {
                let base = (n * channels + c) * out_len;
                let x_base = (n * channels + c) * seq_len;
                for o in 0..out_len {
                    let mut sum = 0.0f64;
                    let mut count = 0usize;
                    for ki in 0..k {
                        let ki_pos = o * stride + ki;
                        // 防 usize 下溢
                        if ki_pos >= padding {
                            let t = ki_pos - padding;
                            if t < seq_len {
                                sum += xd[x_base + t] as f64;
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        out[base + o] = (sum / count as f64) as f32;
                    }
                }
            }
        }
        Tensor::<f32, Cpu>::from_data(out, [batch, channels, out_len])
            .expect("AvgPool1d forward 输出构造失败")
    }
}
