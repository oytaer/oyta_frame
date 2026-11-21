//! log_softmax_stable：数值稳定的 log_softmax（兼容 API，底层走 Tensor）
//!
//! 数学：log softmax(x) = x - max(x) - log Σ exp(x - max(x))
//! 底层：Tensor.softmax(-1) → Tensor.log() → D::softmax + D::elwise_log Backend kernel

use oyta_tensor::prelude::*;

/// 底层 log_softmax（Tensor.softmax → log，f64 累加器 + Rayon 并行）
pub fn log_softmax_stable(logits: &[f32]) -> Vec<f32> {
    let t = Tensor::<f32, Cpu>::from_data(logits.to_vec(), [logits.len()]).unwrap();
    let sm = t.softmax(-1).expect("softmax 底层调用失败");
    sm.log().data_slice().to_vec()
}

/// 底层 softmax（Tensor.softmax，f64 累加器 + Rayon 并行）
pub fn softmax_stable(logits: &[f32]) -> Vec<f32> {
    let t = Tensor::<f32, Cpu>::from_data(logits.to_vec(), [logits.len()]).unwrap();
    t.softmax(-1).expect("softmax 底层调用失败").data_slice().to_vec()
}
