//! Conv1d 一维卷积（纯 Vec 实现 forward，backward 留给 oyta-autograd）
//!
//! 输入布局：NCHW → [batch, in_c, seq_len]
//! 权重布局：[out_c, in_c, k_size]
//! 偏置：[out_c]
//! 输出：[batch, out_c, out_len]，out_len = (seq_len + 2*padding - k_size) / stride + 1
//!
//! 设计文档 §十一 Backend trait 签名参考（当前 registry 版本无此 API，所以手写 forward）

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};
use crate::types::expert_cache::Conv1dCache;

/// Conv1d 配置参数
#[derive(Debug, Clone)]
pub struct Conv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1dConfig {
    /// 计算输出序列长度
    pub fn out_seq_len(&self, seq_len: usize) -> usize {
        (seq_len + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

/// Conv1d 一维卷积层
pub struct Conv1d {
    pub config: Conv1dConfig,
    /// 权重 [out_c, in_c, k_size]
    weight: Parameter,
    /// 偏置 [out_c]
    bias: Parameter,
    /// forward 缓存（backward 需要原始输入 x）
    cache: Conv1dCache,
}

impl Conv1d {
    /// 构造 Conv1d（Kaiming 均匀分布初始化）
    pub fn new(domain_id: u32, module: &str, config: Conv1dConfig) -> Self {
        let k = config.kernel_size;
        let fan_in = config.in_channels * k;
        let scale = (6.0 / fan_in as f32).sqrt();
        // randn 生成 [out_c, in_c, k_size] 权重 → 乘 scale
        let w_raw = Tensor::<f32, Cpu>::randn([config.out_channels, config.in_channels, k]);
        let w_data: Vec<f32> = w_raw.data_slice().iter().map(|v| v * scale).collect();
        let w_tensor = Tensor::<f32, Cpu>::from_data(w_data, [config.out_channels, config.in_channels, k])
            .expect("Conv1d weight 构造失败");
        let b_tensor = Tensor::<f32, Cpu>::zeros([config.out_channels]);

        Self {
            config,
            weight: build_param(domain_id, module, "weight", w_tensor),
            bias:   build_param(domain_id, module, "bias",   b_tensor),
            cache:  Conv1dCache::default(),
        }
    }

    /// 纯 Vec 实现 Conv1d forward
    ///
    /// x: [batch, in_c, seq_len] → out: [batch, out_c, out_len]
    pub fn forward(&mut self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // Step 1: 读取参数
        let w = self.weight.data.lock().unwrap().clone();
        let b = self.bias.data.lock().unwrap().clone();
        let xd = x.data_slice();
        let wd = w.data_slice();
        let bd = b.data_slice();

        // Step 2: 提取 shape
        let [batch, in_c, seq_len] = x.shape().dims[..] else { panic!("Conv1d forward 需要 3D 输入") };
        let out_c = self.config.out_channels;
        let k = self.config.kernel_size;
        let stride = self.config.stride;
        let padding = self.config.padding;
        let out_len = self.config.out_seq_len(seq_len);

        // Step 3: 分配输出 buffer [batch, out_c, out_len]
        let mut out = vec![0.0f32; batch * out_c * out_len];

        // Step 4: NCHW 一维卷积三重循环
        // out[n, oc, o] = bias[oc] + Σ_{ic, k} w[oc, ic, k] * x[n, ic, o*stride + k - padding]
        for n in 0..batch {
            for oc in 0..out_c {
                let base = (n * out_c + oc) * out_len;
                for o in 0..out_len {
                    // 累加器用 f64 防精度损失
                    let mut acc = bd[oc] as f64;
                    for ic in 0..in_c {
                        for ki in 0..k {
                            let ki_pos = o * stride + ki;
                            // 防 usize 下溢：ki_pos < padding 意味着越界（负值位置）
                            if ki_pos >= padding {
                                let t = ki_pos - padding;
                                if t < seq_len {
                                    let x_idx = (n * in_c + ic) * seq_len + t;
                                    let w_idx = (oc * in_c + ic) * k + ki;
                                    acc += (xd[x_idx] as f64) * (wd[w_idx] as f64);
                                }
                            }
                        }
                    }
                    out[base + o] = acc as f32;
                }
            }
        }

        // Step 5: 缓存原始输入（backward 需要）
        self.cache.cached_x = Some(x.clone());

        // Step 6: 构造输出 Tensor
        Tensor::<f32, Cpu>::from_data(out, [batch, out_c, out_len])
            .expect("Conv1d forward 输出构造失败")
    }

    /// 获取 forward 缓存（backward 消费）
    pub fn cache(&self) -> &Conv1dCache { &self.cache }

    /// 清空缓存（每轮 backward 后调）
    pub fn clear_cache(&mut self) { self.cache = Conv1dCache::default(); }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
