//! FusionStrategy 枚举 + TaskHead 配置结构体
//!
//! 纯数据定义（types 层禁止依赖 traits/utils/impls）

/// 融合策略枚举（对齐设计文档 §八 四层分层）
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// 加权求和：Σ gating_weight[i] × expert_output[i]
    /// 无可训练参数，纯数学运算
    WeightedSum,
    /// 拼接 + Linear 投影：concat 激活专家输出 → Linear → fused
    /// 可训练：Linear.weight + Linear.bias
    ConcatLinear { proj_dim: usize },
    /// CrossAttention：Query=占位 CLS token, K/V=专家输出 → FlashAttention → Linear → fused
    /// 可训练：q_proj + kv_proj + out_proj 三组 Linear
    CrossAttention { num_heads: usize, dropout: f32, proj_dim: usize },
}
