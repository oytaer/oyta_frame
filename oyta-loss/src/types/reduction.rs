//! Reduction 枚举：损失函数的归约方式
//!
//! Mean → 对 batch 求平均值（默认，最常用）
//! Sum  → 对 batch 求和
//! None → 每个样本独立返回（[batch] shape）

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// 对 batch 维度求平均（最常用）
    Mean,
    /// 对 batch 维度求和
    Sum,
    /// 每个样本独立返回（Per-Sample Loss）
    None,
}
