//! BatchNorm1d 批归一化（训练态 + 推理态 + 缓存）
//!
//! 训练态：计算 batch 统计 → 归一化 → γ/β 缩放 → 更新 running 统计（指数移动平均）
//! 推理态：只读 running 统计，不更新
//!
//! forward 必须写入缓存 cached_xhat + cached_inv_std（backward 高效计算必需）
//!
//! 设计文档 §十一 BatchNorm1dOps Backend trait 签名参考

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};
use crate::types::expert_cache::BatchNormCache;

/// BatchNorm1d 层
pub struct BatchNorm1d {
    /// 特征维度
    num_features: usize,
    /// 数值稳定常数
    eps: f32,
    /// 动量（running 统计更新速度，默认 0.1）
    momentum: f32,
    /// 缩放参数 gamma（可训练，初始化为 1）
    gamma: Parameter,
    /// 偏移参数 beta（可训练，初始化为 0）
    beta: Parameter,
    /// running mean（推理态用，不可训练，外部可观察）
    pub running_mean: Vec<f32>,
    /// running var（推理态用，不可训练，外部可观察）
    pub running_var: Vec<f32>,
    /// forward 缓存（backward 必需）
    cache: BatchNormCache,
}

impl BatchNorm1d {
    /// 构造 BatchNorm1d（gamma=1, beta=0, running_mean=0, running_var=1）
    pub fn new(domain_id: u32, module: &str, num_features: usize) -> Self {
        let g = Tensor::<f32, Cpu>::from_data(vec![1.0f32; num_features], [num_features])
            .expect("BatchNorm gamma 构造失败");
        let b = Tensor::<f32, Cpu>::zeros([num_features]);
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: build_param(domain_id, module, "gamma", g),
            beta:  build_param(domain_id, module, "beta",  b),
            running_mean: vec![0.0f32; num_features],
            running_var:  vec![1.0f32; num_features],
            cache: BatchNormCache::default(),
        }
    }

    /// BatchNorm forward（自动判断训练态/推理态）
    pub fn forward(&mut self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        if oyta_tensor::is_training_mode() {
            self.train_forward(x)
        } else {
            self.eval_forward(x)
        }
    }

    /// 训练态 forward：计算 batch 统计 + 归一化 + 更新 running 统计 + 写入缓存
    fn train_forward(&mut self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let [batch, num_features] = x.shape().dims[..] else { panic!("BatchNorm1d 需要 2D 输入 [batch, num_features]") };
        let xd = x.data_slice();
        let gd = self.gamma.data.lock().unwrap();
        let bd = self.beta.data.lock().unwrap();
        let gd = gd.data_slice();
        let bd = bd.data_slice();

        // Step 1: 计算 batch 均值 per feature
        let mut batch_mean = vec![0.0f64; num_features];
        for n in 0..batch {
            for f in 0..num_features {
                let v = xd[n * num_features + f] as f64;
                batch_mean[f] += v;
            }
        }
        for f in 0..num_features {
            batch_mean[f] /= batch as f64;
        }

        // Step 2: 计算 batch 方差 per feature
        let mut batch_var = vec![0.0f64; num_features];
        for n in 0..batch {
            for f in 0..num_features {
                let v = xd[n * num_features + f] as f64;
                let diff = v - batch_mean[f];
                batch_var[f] += diff * diff;
            }
        }
        for f in 0..num_features {
            batch_var[f] /= batch as f64;
        }

        // Step 3: 计算 inv_std = 1 / sqrt(var + eps)
        let mut inv_std = vec![0.0f32; num_features];
        for f in 0..num_features {
            inv_std[f] = 1.0 / (batch_var[f] + self.eps as f64).sqrt() as f32;
        }

        // Step 4: 归一化 x̂ = (x - mean) * inv_std，同时输出 y = γ * x̂ + β
        let mut xhat_data = vec![0.0f32; batch * num_features];
        let mut out = vec![0.0f32; batch * num_features];
        for n in 0..batch {
            for f in 0..num_features {
                let x_val = xd[n * num_features + f];
                let x_mean = batch_mean[f] as f32;
                let x_hat = (x_val - x_mean) * inv_std[f];
                xhat_data[n * num_features + f] = x_hat;
                out[n * num_features + f] = gd[f] * x_hat + bd[f];
            }
        }

        // Step 5: 更新 running 统计（指数移动平均）
        // running = (1 - momentum) * running + momentum * batch_value
        for f in 0..num_features {
            self.running_mean[f] = (1.0 - self.momentum) * self.running_mean[f] + self.momentum * batch_mean[f] as f32;
            self.running_var[f]  = (1.0 - self.momentum) * self.running_var[f]  + self.momentum * batch_var[f]  as f32;
        }

        // Step 6: 写入 forward 缓存（backward 必需）
        let xhat_tensor = Tensor::<f32, Cpu>::from_data(xhat_data, [batch, num_features])
            .expect("BN xhat 缓存构造失败");
        let inv_std_tensor = Tensor::<f32, Cpu>::from_data(inv_std, [num_features])
            .expect("BN inv_std 缓存构造失败");
        self.cache.cached_xhat = Some(xhat_tensor);
        self.cache.cached_inv_std = Some(inv_std_tensor);

        Tensor::<f32, Cpu>::from_data(out, [batch, num_features]).expect("BN 训练态输出构造失败")
    }

    /// 推理态 forward：只读 running 统计，不更新，不写缓存
    fn eval_forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let [batch, num_features] = x.shape().dims[..] else { panic!("BatchNorm1d 需要 2D 输入") };
        let xd = x.data_slice();
        let gd = self.gamma.data.lock().unwrap();
        let bd = self.beta.data.lock().unwrap();
        let gd = gd.data_slice();
        let bd = bd.data_slice();

        let mut out = vec![0.0f32; batch * num_features];
        for n in 0..batch {
            for f in 0..num_features {
                let x_val = xd[n * num_features + f];
                let inv_std = 1.0 / (self.running_var[f] + self.eps).sqrt();
                let x_hat = (x_val - self.running_mean[f]) * inv_std;
                out[n * num_features + f] = gd[f] * x_hat + bd[f];
            }
        }
        Tensor::<f32, Cpu>::from_data(out, [batch, num_features]).expect("BN 推理态输出构造失败")
    }

    pub fn cache(&self) -> &BatchNormCache { &self.cache }
    pub fn clear_cache(&mut self) { self.cache = BatchNormCache::default(); }

    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    pub fn num_features(&self) -> usize { self.num_features }
}
