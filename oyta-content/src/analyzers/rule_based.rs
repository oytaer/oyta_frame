//! RuleBasedAnalyzer：纯 regex 关键词匹配 + 结构化模式 + metadata 辅助路由
//!
//! 架构文档 §5.2 三种分析维度全部实现：
//!   ✅ 关键词/实体匹配 — 预编译 regex HashMap（构造时一次性编译）
//!   ✅ 句法/结构特征   — SyntaxPatternLibrary（数字+元/数字+/+数字 等结构化模式）
//!   ✅ 元数据辅助路由   — MetaRouter（扩展名/来源 URL/时间戳/作者）
//!
//! 完整流程：
//!   1. 关键词 hits/ln(len) → 静态得分
//!   2. SyntaxPatternLibrary.detect → 结构化加分（金融-数字+点/医疗-数字/数字/法律-第X条...）
//!   3. 两者加权融合（静态权重 0.7 + 结构化权重 0.3）
//!   4. MetaRouter.adjust(weights, metadata, timestamp) → metadata 辅助调整
//!   5. 归一化 + 全零 fallback + 输出 FeatureEncoder Tensor

use crate::traits::analyzer::ContentAnalyzer;
use crate::types::domain::Domain;
use crate::types::raw_input::RawInput;
use crate::types::content_analysis::{ContentAnalysis, Evidence};
use crate::utils::feature_encoder::FeatureEncoder;
use crate::utils::padding_mask::PaddingMask;
use crate::utils::syntax_pattern::SyntaxPatternLibrary;
use crate::utils::meta_router::MetaRouter;
use std::collections::HashMap;
use regex::Regex;

/// RuleBasedAnalyzer 融合配置
#[derive(Debug, Clone)]
pub struct RuleBasedConfig {
    /// 关键词匹配在融合里的权重（默认 0.7）
    pub keyword_weight: f32,
    /// 结构化模式在融合里的权重（默认 0.3）
    pub syntax_weight: f32,
}

impl Default for RuleBasedConfig {
    fn default() -> Self { Self { keyword_weight: 0.7, syntax_weight: 0.3 } }
}

pub struct RuleBasedAnalyzer {
    /// 所有注册领域
    domains: Vec<Domain>,
    /// 预编译 regex：domain_id → Vec<Regex>（关键词词典）
    keyword_regexes: HashMap<u32, Vec<Regex>>,
    /// 结构化模式库（预编译的领域特定正则）
    syntax_lib: SyntaxPatternLibrary,
    /// metadata + timestamp 辅助路由器
    meta_router: MetaRouter,
    /// 特征编码器（hash n-gram → Tensor，可选 MiniMLP）
    feature_encoder: FeatureEncoder,
    /// 融合配置
    config: RuleBasedConfig,
}

impl RuleBasedAnalyzer {
    pub fn new(domains: Vec<Domain>) -> Self {
        Self::with_config(domains, RuleBasedConfig::default(), 256)
    }

    pub fn with_config(domains: Vec<Domain>, config: RuleBasedConfig, feature_dim: usize) -> Self {
        // 关键词 regex 预编译（一次性，避免每次 analyze 重新编译）
        let mut keyword_regexes: HashMap<u32, Vec<Regex>> = HashMap::new();
        for domain in &domains {
            let regexes: Vec<Regex> = domain.keywords.iter()
                .filter_map(|kw| Regex::new(&format!(r"(?i)(?:{})", regex::escape(kw))).ok())
                .collect();
            keyword_regexes.insert(domain.id, regexes);
        }
        Self {
            domains,
            keyword_regexes,
            syntax_lib: SyntaxPatternLibrary::new(),
            meta_router: MetaRouter::new(),
            feature_encoder: FeatureEncoder::new(feature_dim),
            config,
        }
    }

    /// 内部辅助：统计关键词命中次数
    fn count_hits(text: &str, regexes: &[Regex]) -> usize {
        regexes.iter().map(|re| re.find_iter(text).count()).sum()
    }

    /// 内部辅助：关键词 hits → 静态权重
    fn keyword_weights(&self, text: &str) -> Vec<(u32, f32)> {
        let total_chars = text.chars().count().max(1) as f32;
        let char_ln = total_chars.ln().max(1.0);
        self.domains.iter().map(|d| {
            let hits = Self::count_hits(text, self.keyword_regexes.get(&d.id).unwrap());
            (d.id, (hits as f32 / char_ln).min(1.0))
        }).collect()
    }

    /// 内部辅助：多源权重融合（关键词 × 0.7 + 结构化 × 0.3）
    fn fuse_keyword_syntax(keyword_w: &[(u32, f32)], syntax_w: &[(u32, f32)],
                           kw_r: f32, sy_r: f32) -> Vec<(u32, f32)> {
        let mut sy_map: HashMap<u32, f32> = syntax_w.iter().copied().collect();
        keyword_w.iter().map(|(id, kw)| {
            let sy = sy_map.remove(id).unwrap_or(0.0);
            (*id, kw * kw_r + sy * sy_r)
        }).collect()
    }
}

impl ContentAnalyzer for RuleBasedAnalyzer {
    fn analyze(&self, raw_input: &RawInput) -> Result<ContentAnalysis, Box<dyn std::error::Error>> {
        let text = &raw_input.text;

        // ===== Step 1：关键词 hits → 静态权重 =====
        let keyword_w = self.keyword_weights(text);

        // ===== Step 2：SyntaxPatternLibrary → 结构化加分 =====
        let syntax_w = self.syntax_lib.detect(text);

        // ===== Step 3：融合（关键词 × 0.7 + 结构化 × 0.3）=====
        let mut raw_weights = Self::fuse_keyword_syntax(
            &keyword_w, &syntax_w, self.config.keyword_weight, self.config.syntax_weight
        );

        // ===== Step 4：全零 fallback（一个关键词都没命中也没结构化信号时）=====
        let total_w: f32 = raw_weights.iter().map(|(_, w)| *w).sum();
        let best_id = raw_weights.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(id, _)| *id).unwrap_or(0);
        if total_w <= 0.0 {
            raw_weights = raw_weights.iter().map(|(id, _)| {
                if *id == best_id { (*id, 1.0) } else { (*id, 0.0) }
            }).collect();
        }

        // ===== Step 5：MetaRouter.adjust → metadata/timestamp 辅助权重调整 =====
        let weights = self.meta_router.adjust(&raw_weights, &raw_input.metadata, raw_input.timestamp);

        // ===== Step 6：归一化 + 生成 Evidence =====
        let total: f32 = weights.iter().map(|(_, w)| *w).sum();
        let mut normalized: Vec<(u32, f32)> = if total > 0.0 {
            weights.iter().map(|(id, w)| (*id, w / total)).collect()
        } else { weights.clone() };
        normalized.retain(|(_, w)| *w > 0.0);

        // Evidence：关键词命中 + 结构化命中的可解释性
        let mut evidence: Vec<Evidence> = Vec::new();
        for (did, score) in &keyword_w {
            if *score > 0.0 {
                let domain = self.domains.iter().find(|d| d.id == *did);
                let name = domain.map(|d| d.name.as_str()).unwrap_or("未知");
                evidence.push(Evidence {
                    domain_id: *did,
                    description: format!("关键词命中（领域：{}）", name),
                    contribution: *score,
                });
            }
        }
        for (did, score) in &syntax_w {
            if *score > 0.0 {
                let domain = self.domains.iter().find(|d| d.id == *did);
                let name = domain.map(|d| d.name.as_str()).unwrap_or("未知");
                evidence.push(Evidence {
                    domain_id: *did,
                    description: format!("结构化模式命中（领域：{}，加分 {}）", name, score),
                    contribution: *score * self.config.syntax_weight,
                });
            }
        }

        // ===== Step 7：FeatureEncoder → 固定维度 Tensor =====
        let feature_vector = self.feature_encoder.encode(text, &self.domains);

        // ===== Step 8：变长输入 padding mask =====
        let mut analysis = ContentAnalysis::new(normalized, feature_vector);
        analysis.evidence = evidence;
        if let Some(ref lengths) = raw_input.sample_lengths {
            let max_len = raw_input.max_padded_length
                .unwrap_or_else(|| *lengths.iter().max().unwrap_or(&256));
            analysis = analysis.with_padding_mask(PaddingMask::detect(lengths, max_len));
        }

        Ok(analysis)
    }

    fn domains(&self) -> &[Domain] { &self.domains }

    fn rule_based_match(&self, text: &str) -> Vec<(u32, f32)> {
        // 返回纯关键词匹配（不带结构化/metadata，HybridAnalyzer 硬规则判断用）
        self.keyword_weights(text)
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .collect()
    }

    fn train_semantic_encoder(&mut self, _training_data: &[crate::traits::analyzer::LabeledContent]) {
        // RuleBasedAnalyzer 不训练（纯规则，FeatureEncoder 是静态 hash）
    }

    fn feature_dim(&self) -> usize { self.feature_encoder.effective_output_dim() }
}
