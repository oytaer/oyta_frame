//! OpType：算子类型枚举（避免 router → autograd 循环依赖）
//!
//! 三个掩码变体的含义（架构 0.6.8）：
//!   Conv1dMasked           —— 带通道级 mask（某些输出通道被 mask 掉）
//!   Conv1dWithSequenceMask —— 带序列级 mask（变长/padding 位置不参与卷积）
//!   BatchNorm1dMasked      —— 带通道级 mask（某些特征跳过 BN 归一化）
//!
//! router crate 只持有这个枚举，autograd crate 根据 OpType 分发 backward 逻辑

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    // ===== Phase 0 新增：卷积/池化 =====
    Conv1d,
    Conv1dMasked,              // 通道级 mask
    Conv1dWithSequenceMask,    // 序列级 mask（padding）
    BatchNorm1d,
    BatchNorm1dMasked,         // 通道级 mask
    MaxPool1d,
    AvgPool1d,

    // ===== oyta-tensor 已有 =====
    Linear,
    Matmul,
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    Silu,
    Softmax,
    LayerNorm,
    RMSNorm,
    FlashAttention,
}

impl OpType {
    /// 是否是带掩码的变体（router 层不直接用，autograd 层 resolve_op_mask 用）
    pub fn is_masked_variant(&self) -> bool {
        matches!(self,
            Self::Conv1dMasked | Self::Conv1dWithSequenceMask | Self::BatchNorm1dMasked
        )
    }

    /// 掩码变体对应的基础 OpType（resolve_op_mask 时查原始 Op）
    pub fn base_variant(&self) -> Self {
        match self {
            Self::Conv1dMasked | Self::Conv1dWithSequenceMask => Self::Conv1d,
            Self::BatchNorm1dMasked => Self::BatchNorm1d,
            other => *other,
        }
    }
}
