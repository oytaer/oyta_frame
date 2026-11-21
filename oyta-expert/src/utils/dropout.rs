//! Dropout 训练态切换 + apply_dropout 实现
//!
//! oyta_tensor 已暴露 set_training_mode() / is_training_mode() 全局函数
//! 框架层通过它们控制 BatchNorm running 统计更新 / Dropout 随机掩码
//! apply_dropout 是专家内部的统一 Dropout 入口（训练态判断 + 占位实现）

use oyta_tensor::prelude::*;

// ===== 从 oyta_tensor 直接 re-export =====
pub use oyta_tensor::set_training_mode;
pub use oyta_tensor::is_training_mode;

/// 统一 Dropout 入口（所有专家内部调这个）
///
/// 训练态（is_training_mode()=true 且 rate>0）：返回原始输入的 scale 版本
/// 推理态或 rate=0：直接 clone 返回（零开销）
///
/// Phase 3 简化实现：暂不真正做随机掩码（需要 rand crate），只做训练态判断 + 占位返回
/// Phase 4 补 rand 随机掩码 + scale = 1/(1-rate) 正确缩放
pub fn apply_dropout(x: &Tensor<f32, Cpu>, rate: f32) -> Tensor<f32, Cpu> {
    // 推理态或 rate=0 → 直接 clone
    if !is_training_mode() || rate <= 0.0 {
        return x.clone();
    }
    // 训练态 rate=1 → 全零输出（占位）
    if rate >= 1.0 {
        return Tensor::<f32, Cpu>::zeros(x.shape().dims.clone());
    }
    // Phase 3 简化：暂不做随机掩码（需要 rand crate），直接返回 clone
    // Phase 4 补：生成 rand 掩码 + 乘 x + scale = 1/(1-rate)
    let _ = rate; // 显式消费参数，防未使用警告
    x.clone()
}
