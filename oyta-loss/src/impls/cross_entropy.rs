//! CrossEntropyLoss：Softmax + NLL
//!
//! 数学原理：
//!   loss = -Σ target[c] × log softmax(logits[c])
//!   grad = softmax(logits) - one_hot(target)  ← 一行解析梯度
//!
//! label_smoothing 支持：把硬标签平滑成软标签
//!   target_smooth = (1-ε) × one_hot + ε/(K-1) × uniform
//!   → 防止模型过度自信（泛化能力提升）
//!
//! 底层调用链（100% 走 oyta-tensor Backend trait）：
//!   Tensor.softmax(-1) → D::softmax     (f64 累加器 + Rayon 并行)
//!   Tensor.log()        → D::elwise_log (一元算子 Backend kernel)
//!   Tensor.mean(None)   → D::reduce_mean (归约 Backend kernel)
//!   Tensor.sum()        → reduce_sum     (归约)

use oyta_tensor::prelude::*;
use crate::traits::loss_fn::LossFn;
use crate::types::reduction::Reduction;

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
    /// 前向损失计算
    /// 底层：softmax → log → per-sample NLL → reduce
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0];
        let nc = self.num_classes;
        let td = target.data_slice();

        // ===== 底层 softmax（f64 累加器数值稳定 + Rayon 并行）=====
        let sm_tensor = logits.softmax(-1).expect("softmax 底层调用失败");
        // ===== 底层 log（D::elwise_log Backend kernel）=====
        let log_sm_tensor = sm_tensor.log();
        let log_sm = log_sm_tensor.data_slice();  // &[f32]

        // ===== 计算 per-sample NLL（逐样本逻辑，难以 Tensor 化）=====
        let mut per_sample: Vec<f32> = Vec::with_capacity(batch);
        let eps = self.label_smoothing;
        for n in 0..batch {
            let row = n * nc;
            let mut loss_nll: f64 = 0.0;
            if target.numel() == batch {
                let class = td[n] as usize;
                if eps > 0.0 {
                    let eps_f64 = eps as f64;
                    let uniform = eps_f64 / (nc as f64 - 1.0).max(1.0);
                    for c in 0..nc {
                        let weight = if c == class { 1.0 - eps_f64 } else { uniform };
                        loss_nll -= weight * (log_sm[row + c] as f64);
                    }
                } else {
                    loss_nll = -(log_sm[row + class] as f64);
                }
            } else {
                for c in 0..nc {
                    loss_nll -= td[n * nc + c] as f64 * (log_sm[row + c] as f64);
                }
            }
            per_sample.push(loss_nll as f32);
        }

        // ===== 底层 Reduction（D::reduce_mean / reduce_sum Backend kernel）=====
        let per_tensor = Tensor::<f32, Cpu>::from_data(per_sample, [batch]).unwrap();
        match self.reduction {
            Reduction::Mean => {
                // 底层 mean(None) → D::reduce_mean
                let m = per_tensor.mean(None).expect("mean 底层调用失败");
                Tensor::<f32, Cpu>::from_data(vec![m.item()], [1]).unwrap()
            }
            Reduction::Sum => {
                // 底层 sum() → reduce_sum
                let s: f32 = per_tensor.sum().expect("sum 底层调用失败");
                Tensor::<f32, Cpu>::from_data(vec![s], [1]).unwrap()
            }
            Reduction::None => per_tensor,
        }
    }

    /// 解析梯度：grad = softmax(logits) - one_hot(target)
    /// 底层：softmax → Vec one_hot 构造 → Tensor
    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let batch = logits.shape().dims[0];
        let nc = self.num_classes;
        let td = target.data_slice();

        // ===== 底层 softmax =====
        let sm_tensor = logits.softmax(-1).expect("softmax 底层调用失败");
        let sm = sm_tensor.data_slice();

        // grad = softmax - one_hot
        let mut grad = vec![0.0f32; batch * nc];
        for n in 0..batch {
            let row = n * nc;
            if target.numel() == batch {
                let class = td[n] as usize;
                for c in 0..nc {
                    grad[row + c] = sm[row + c] - if c == class { 1.0 } else { 0.0 };
                }
            } else {
                for c in 0..nc { grad[row + c] = sm[row + c] - td[n * nc + c]; }
            }
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
