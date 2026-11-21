//! CrossEntropyLoss：Softmax + NLL（log_softmax 数值稳定版本）
//!
//! 数学原理：
//!   loss = -Σ target[c] × log softmax(logits[c])
//!   grad = softmax(logits) - one_hot(target)  ← 一行解析梯度
//!
//! label_smoothing 支持：把硬标签平滑成软标签
//!   target_smooth = (1-ε) × one_hot + ε/(K-1) × uniform
//!   → 防止模型过度自信（泛化能力提升）

use oyta_tensor::prelude::*;
use crate::traits::loss_fn::LossFn;
use crate::types::reduction::Reduction;
use crate::utils::log_softmax::{log_softmax_stable, softmax_stable};

/// 交叉熵损失函数
pub struct CrossEntropyLoss {
    /// 分类数量（logits 输出维度）
    pub num_classes: usize,
    /// 归约方式（Mean / Sum / None）
    pub reduction: Reduction,
    /// 标签平滑系数（0.0 = 硬标签，常用 0.1）
    pub label_smoothing: f32,
}

impl CrossEntropyLoss {
    /// 构造 CrossEntropyLoss（label_smoothing 默认 0.0）
    pub fn new(num_classes: usize, reduction: Reduction) -> Self {
        Self { num_classes, reduction, label_smoothing: 0.0 }
    }

    /// 设置标签平滑系数（返回 self 支持链式调用）
    pub fn with_label_smoothing(mut self, eps: f32) -> Self {
        self.label_smoothing = eps;
        self
    }
}

impl LossFn for CrossEntropyLoss {
    /// 前向损失计算（数值稳定 log_softmax）
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0];
        let nc = self.num_classes;
        let ld = logits.data_slice();
        let td = target.data_slice();

        let mut per_sample: Vec<f32> = Vec::with_capacity(batch);
        for n in 0..batch {
            let row = n * nc;
            // ===== 数值稳定 log_softmax =====
            let logits_row: Vec<f32> = (0..nc).map(|c| ld[row + c]).collect();
            let log_sm = log_softmax_stable(&logits_row);

            // ===== 构造 target（支持 class index 和 one-hot 两种格式）=====
            // target.numel() == batch → 每个元素是 class index
            // target.numel() == batch * nc → one-hot 分布
            let eps = self.label_smoothing;
            let mut loss_nll: f64 = 0.0;
            if target.numel() == batch {
                let class = td[n] as usize;
                // label_smoothing: 硬标签 → 软标签
                // loss = -(1-ε) × log_sm[class] - Σ_{c≠class} ε/(K-1) × log_sm[c]
                if eps > 0.0 {
                    let eps_f64 = eps as f64;
                    let uniform = eps_f64 / (nc as f64 - 1.0).max(1.0);
                    for c in 0..nc {
                        let weight = if c == class { 1.0 - eps_f64 } else { uniform };
                        loss_nll -= weight * (log_sm[c] as f64);
                    }
                } else {
                    loss_nll = -(log_sm[class] as f64);
                }
            } else {
                // one-hot / 软标签分布
                for c in 0..nc {
                    loss_nll -= td[n * nc + c] as f64 * (log_sm[c] as f64);
                }
            }
            per_sample.push(loss_nll as f32);
        }

        // ===== Reduction =====
        match self.reduction {
            Reduction::Mean => {
                let m = per_sample.iter().sum::<f32>() / per_sample.len() as f32;
                Tensor::<f32, Cpu>::from_data(vec![m], [1]).unwrap()
            }
            Reduction::Sum => {
                let s: f32 = per_sample.iter().sum();
                Tensor::<f32, Cpu>::from_data(vec![s], [1]).unwrap()
            }
            Reduction::None => {
                Tensor::<f32, Cpu>::from_data(per_sample, [batch]).unwrap()
            }
        }
    }

    /// 解析梯度：grad = softmax(logits) - one_hot(target)
    ///
    /// 一行精确梯度（PyTorch 内部实现）
    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0];
        let nc = self.num_classes;
        let ld = logits.data_slice();
        let td = target.data_slice();
        let mut grad = vec![0.0f32; batch * nc];

        for n in 0..batch {
            let row = n * nc;
            // softmax
            let logits_row: Vec<f32> = (0..nc).map(|c| ld[row + c]).collect();
            let sm = softmax_stable(&logits_row);

            // one_hot
            let mut one_hot = vec![0.0f32; nc];
            if target.numel() == batch {
                one_hot[td[n] as usize] = 1.0;
            } else {
                for c in 0..nc { one_hot[c] = td[n * nc + c]; }
            }

            // grad = softmax - one_hot
            for c in 0..nc { grad[row + c] = sm[c] - one_hot[c]; }
        }

        // Reduction 缩放
        if matches!(self.reduction, Reduction::Mean) {
            let scale = 1.0 / batch as f32;
            for g in grad.iter_mut() { *g *= scale; }
        }
        Tensor::<f32, Cpu>::from_data(grad, [batch, nc]).unwrap()
    }

    fn reduction(&self) -> Reduction { self.reduction }
}
