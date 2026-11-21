//! ContentAnalyzer trait：所有内容分析器基类
//!
//! 三种内置实现（全部完整实现，无骨架无占位）：
//!   RuleBasedAnalyzer —— regex 关键词 + 句法结构化模式 + metadata 辅助路由
//!   TFIDFAnalyzer     —— TF-IDF cosine 相似度 + 句法结构化模式 + metadata 辅助路由
//!   HybridAnalyzer    —— 规则优先 + TFIDF 融合 + MiniMLP 监督训练（推荐默认）
//!
//! trait 约束 Send + Sync：支持 rayon 多线程并行分析批量输入

use crate::types::domain::Domain;
use crate::types::raw_input::RawInput;
use crate::types::content_analysis::ContentAnalysis;
use std::error::Error;

/// 标注内容（HybridAnalyzer.train_semantic_encoder 的监督训练数据）
#[derive(Debug, Clone)]
pub struct LabeledContent {
    /// 原始输入
    pub raw: RawInput,
    /// 人工标注的领域权重（一个样本可以属于多个领域）
    pub true_weights: Vec<(u32, f32)>,
}

/// 内容分析器基类 trait
pub trait ContentAnalyzer: Send + Sync {
    /// ===== 主入口方法：分析完整内容 =====
    /// 输入：原始输入文本 + 可选结构化数据 + 可选变长信息
    /// 输出：领域软权重 + 固定维度特征向量 + 命中证据 + padding mask（如果变长）
    fn analyze(&self, raw_input: &RawInput) -> Result<ContentAnalysis, Box<dyn Error>>;

    /// 返回所有已注册领域（DomainRouter 用）
    fn domains(&self) -> &[Domain];

    /// ===== 硬规则快速匹配 =====
    /// RuleBasedAnalyzer / HybridAnalyzer 必须实现
    /// 返回 [(domain_id, raw_weight)] 形式的关键词命中结果
    /// TFIDFAnalyzer 返回空 Vec（纯向量分析，无硬规则）
    fn rule_based_match(&self, _text: &str) -> Vec<(u32, f32)> { Vec::new() }

    /// 可选：训练轻量语义编码器（提升路由精度）
    /// HybridAnalyzer 有完整实现：MiniMLP 手动反向传播 + SGD 更新
    /// RuleBasedAnalyzer / TFIDFAnalyzer 默认空实现（纯规则，不训练）
    fn train_semantic_encoder(&mut self, _training_data: &[LabeledContent]) {}

    /// 输出特征向量的维度（FeatureEncoder 内部固定）
    fn feature_dim(&self) -> usize;
}
