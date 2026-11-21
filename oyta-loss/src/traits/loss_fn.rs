//! LossFn trait：损失函数统一接口
//!
//! 所有损失函数（CrossEntropy / MSE / ...）都实现此 trait
//! compute = 前向损失值，grad = 解析梯度（dL/dlogits）

use oyta_tensor::prelude::*;
use crate::types::reduction::Reduction;

/// 损失函数统一 trait（Send + Sync → 可跨线程共享）
pub trait LossFn: Send + Sync {
    /// 计算前向损失值
    ///
    /// # 参数
    /// - `logits`: 模型输出（未归一化的原始分数）
    /// - `target`: 标签（class index 或 one-hot）
    ///
    /// # 返回
    /// 归约后的损失 Tensor（shape 取决于 Reduction）
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;

    /// 计算解析梯度（dL/dlogits）
    ///
    /// CrossEntropy 的梯度 = softmax - one_hot（一行代码，高效精确）
    /// MSE 的梯度 = (2/N) * (pred - target)
    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;

    /// 把 loss Tensor 转 f32 标量（item 函数，PyTorch 风格）
    ///
    /// - numel=1 → 直接返回 data[0]
    /// - numel>1 → 对所有元素取平均
    fn item(&self, loss: &Tensor<f32, Cpu>) -> f32 {
        if loss.numel() == 1 { return loss.data_slice()[0]; }
        let d = loss.data_slice();
        d.iter().sum::<f32>() / d.len() as f32
    }

    /// 返回当前 Reduction 方式
    fn reduction(&self) -> Reduction;
}
