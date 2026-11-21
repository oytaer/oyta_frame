//! Op 枚举：计算图节点的算子类型（纯数据，types 层）
//!
//! 四层分层约束：types 层不依赖 impls 层 → Op 只定义枚举，backward 函数在 impls/op.rs
//!
//! 变体分三类：
//! 1. 标准算子（Phase 1 基础算子，backward 底层调 oyta-tensor）
//! 2. Phase 2 掩码变体（对齐 oyta-router OpType，架构 0.6.8 三层稀疏）
//! 3. Loss 算子（交叉熵、MSE 等）

/// Op 枚举：计算图节点的算子类型
#[derive(Debug, Clone)]
pub enum Op {
    // ===== 叶子 =====
    Leaf,

    // ===== Phase 1：标准算子 =====
    Linear,                              // input: [x, w, b] → [x @ w^T + b]
    Matmul { transa: bool, transb: bool },
    Relu, Gelu, Sigmoid, Tanh, Silu,     // 激活函数（纯逐元素）
    Softmax { dim: isize },
    LayerNorm { eps: f32 },
    RMSNorm { eps: f32 },
    Conv1d { kernel_size: usize, stride: usize, padding: usize }, // 标准卷积
    BatchNorm1d { eps: f32, momentum: f32 },                       // 标准 BN
    MaxPool1d { kernel_size: usize, stride: usize },              // 标准 MaxPool

    // ===== Phase 2：带掩码的稀疏变体（架构 0.6.8 精确定义）=====
    /// Conv1d 带通道级 mask（L2 稀疏）
    Conv1dMasked { channel_mask: Vec<f32> },
    /// BatchNorm1d 带通道级 mask（L2 稀疏）
    BatchNorm1dMasked { channel_mask: Vec<f32> },
    /// Conv1d 带序列级 mask（L3 稀疏，padding 时间步）
    Conv1dWithSequenceMask { sequence_mask: Vec<f32> },
    /// Router 算子（不可训练，backward 返回空 Vec）
    Router { domain_mask: Vec<bool>, normalized_weights: Vec<f32> },

    // ===== Loss =====
    CrossEntropy { num_classes: usize },
    MSE,
}

impl Op {
    /// 返回该 Op 所属的领域 ID（可选）
    ///
    /// Router / Loss / TaskHead → None
    /// Expert 内部算子 → Some(did)（Op 自身不存储 domain_id，由调用方传入）
    /// 辅助查询：帮助框架层做稀疏路由决策
    pub fn domain_id(&self) -> Option<u32> {
        match self {
            // 不可训练 / 全局算子 → None
            Op::Router { .. } | Op::Leaf | Op::MSE | Op::CrossEntropy { .. } => None,
            // 其他算子默认 None（domain_id 实际存储在 Node 上）
            _ => None,
        }
    }

    /// 是否是掩码变体（用于 sparse_backward 三层过滤分发）
    pub fn is_masked_variant(&self) -> bool {
        matches!(
            self,
            Op::Conv1dMasked { .. }
                | Op::BatchNorm1dMasked { .. }
                | Op::Conv1dWithSequenceMask { .. }
        )
    }

    /// 是否可训练（需要 backward 计算梯度）
    pub fn is_trainable(&self) -> bool {
        !matches!(self, Op::Router { .. } | Op::Leaf)
    }
}
