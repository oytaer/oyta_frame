//! ContentAnalysis：ContentAnalyzer 的统一输出
//!
//! 两个核心输出：
//!   1. domain_weights  —— 各领域软分配权重（Router 用它做稀疏决策）
//!   2. feature_vector  —— 固定维度特征 Tensor（所有激活专家统一输入）
//!
//! 附加输出：
//!   evidence     —— 命中证据列表（可解释性 + debug）
//!   padding_mask —— 变长输入的 padding mask（Conv1d 内部 L3 稀疏用）

use oyta_tensor::prelude::*;

/// 命中证据：说明某个领域为什么获得了某个权重
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    /// 被命中的领域 ID
    pub domain_id: u32,
    /// 人类可读的证据描述（如 "匹配到关键词：净利润"）
    pub description: String,
    /// 原始权重贡献（0~1）
    pub contribution: f32,
}

/// ContentAnalyzer 完整输出
pub struct ContentAnalysis {
    /// 各领域软分配权重（不要求和为 1，Router 会再归一化）
    /// Vec<(domain_id, weight)>
    pub domain_weights: Vec<(u32, f32)>,

    /// 固定维度特征 Tensor（所有专家统一输入）
    /// shape = [1, feature_dim]  单样本 batch=1
    pub feature_vector: Tensor<f32, Cpu>,

    /// 命中证据列表（RuleBasedAnalyzer 填充，TFIDF 可能为空）
    pub evidence: Vec<Evidence>,

    /// 变长输入的 padding mask（None = 定长输入）
    /// shape = [num_samples, max_padded_length]
    /// mask[n][t] = 0.0 → 位置 t 有效
    /// mask[n][t] = 1.0 → 位置 t 是 padding
    pub padding_mask: Option<Vec<Vec<f32>>>,
}

impl ContentAnalysis {
    /// 最简构造（只填 weights + feature_vector）
    pub fn new(weights: Vec<(u32, f32)>, fv: Tensor<f32, Cpu>) -> Self {
        Self {
            domain_weights: weights,
            feature_vector: fv,
            evidence: Vec::new(),           // 默认空，RuleBasedAnalyzer.analyze 填充
            padding_mask: None,             // 默认空，变长输入才填
        }
    }

    /// 链式添加命中证据
    pub fn add_evidence(mut self, domain_id: u32, description: impl Into<String>, contribution: f32) -> Self {
        self.evidence.push(Evidence {
            domain_id,
            description: description.into(),
            contribution,
        });
        self
    }

    /// 链式设置 padding mask
    pub fn with_padding_mask(mut self, mask: Vec<Vec<f32>>) -> Self {
        self.padding_mask = Some(mask);
        self
    }
}
