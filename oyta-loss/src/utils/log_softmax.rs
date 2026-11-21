//! log_softmax_stable：数值稳定的 log_softmax（CrossEntropy 共享）
//!
//! 数学：log softmax(x) = x - max(x) - log Σ exp(x - max(x))
//! 减 max 防 exp 溢出（数值稳定核心技巧）

/// 数值稳定的 log_softmax（对单个样本 logit 向量）
///
/// # 参数
/// - `logits`: 原始分数向量（长度 = num_classes）
///
/// # 返回
/// log_softmax 结果向量（同长度）
pub fn log_softmax_stable(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // 防 max_v 是 NaN / Inf 的极端情况
    let max_v = if max_v.is_finite() { max_v } else { 0.0 };
    // f64 累加防精度损失
    let sum_exp: f64 = logits.iter().map(|&x| (x - max_v).exp() as f64).sum();
    let log_sum_exp = sum_exp.ln() as f32 + max_v;
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// 数值稳定的 softmax（对单个样本 logit 向量）
pub fn softmax_stable(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_v = if max_v.is_finite() { max_v } else { 0.0 };
    let sum_exp: f64 = logits.iter().map(|&x| (x - max_v).exp() as f64).sum();
    logits.iter().map(|&x| ((x - max_v).exp() as f64 / sum_exp) as f32).collect()
}
