//! TFIDFAnalyzer：TF-IDF 向量相似度 + 句法模式 + metadata 辅助路由
//!
//! 架构文档 §5.2 三种分析维度全部实现：
//!   ✅ TF-IDF 主题向量 cosine 相似度
//!   ✅ 句法/结构特征（SyntaxPatternLibrary 结构化加分）
//!   ✅ 元数据辅助路由（MetaRouter 扩展名/来源/时间戳/作者）
//!
//! 完整流程：
//!   1. 预构建：每个领域主题向量（TF × IDF → hash 桶 → L2 归一化）
//!   2. analyze：输入文本 → TF-IDF 向量 → cosine 相似度 → softmax
//!   3. SyntaxPattern 结构化加分（和 cosine 加权融合 0.7/0.3）
//!   4. MetaRouter.adjust(weights, metadata, timestamp) → metadata 辅助
//!   5. 归一化 + Evidence + PaddingMask + FeatureEncoder Tensor

use crate::traits::analyzer::ContentAnalyzer;
use crate::types::domain::Domain;
use crate::types::raw_input::RawInput;
use crate::types::content_analysis::{ContentAnalysis, Evidence};
use crate::utils::feature_encoder::FeatureEncoder;
use crate::utils::padding_mask::PaddingMask;
use crate::utils::syntax_pattern::SyntaxPatternLibrary;
use crate::utils::meta_router::MetaRouter;

pub struct TFIDFAnalyzer {
    /// 所有注册领域
    domains: Vec<Domain>,
    /// 每个领域的预构建主题向量（hash n-gram + TF-IDF → 固定维度 Vec<f32>）
    domain_vectors: std::collections::HashMap<u32, Vec<f32>>,
    /// IDF 词典：n-gram → log(num_domains / df)，罕见 n-gram 权重更高
    idf: std::collections::HashMap<String, f32>,
    /// 特征编码器（hash → Tensor，可选 MiniMLP）
    feature_encoder: FeatureEncoder,
    /// 句法/结构特征库（结构化正则预编译）
    syntax_lib: SyntaxPatternLibrary,
    /// metadata + timestamp 辅助路由器
    meta_router: MetaRouter,
    /// 主题向量维度（与 FeatureEncoder 保持一致）
    vector_dim: usize,
}

impl TFIDFAnalyzer {
    pub fn new(domains: Vec<Domain>) -> Self {
        let vector_dim = 256;
        let mut analyzer = Self {
            domains: domains.clone(),
            domain_vectors: std::collections::HashMap::new(),
            idf: std::collections::HashMap::new(),
            feature_encoder: FeatureEncoder::new(vector_dim),
            syntax_lib: SyntaxPatternLibrary::new(),
            meta_router: MetaRouter::new(),
            vector_dim,
        };
        analyzer.build_domain_vectors();
        analyzer
    }

    /// 预构建：每个领域主题向量 + IDF 词典
    fn build_domain_vectors(&mut self) {
        let num_domains = self.domains.len() as f32;
        // Step 1: 统计 n-gram 出现的领域数 → IDF
        let mut ngram_domain_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for domain in &self.domains {
            let mut domain_ngrams: std::collections::HashSet<String> = std::collections::HashSet::new();
            for kw in &domain.keywords {
                for ngram in Self::extract_ngrams(kw) { domain_ngrams.insert(ngram); }
            }
            for ng in domain_ngrams { *ngram_domain_count.entry(ng).or_insert(0) += 1; }
        }
        for (ng, df) in &ngram_domain_count {
            self.idf.insert(ng.clone(), (num_domains / (*df as f32 + 1.0)).ln() + 1.0);
        }
        // Step 2: 每个领域的主题向量（TF × IDF → hash 桶 → L2 归一化）
        for domain in &self.domains {
            let mut theme = vec![0.0f32; self.vector_dim];
            let mut tf_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
            for kw in &domain.keywords {
                for ng in Self::extract_ngrams(kw) { *tf_map.entry(ng).or_insert(0.0) += 1.0; }
            }
            for (ng, tf) in &tf_map {
                let idf = self.idf.get(ng).copied().unwrap_or(1.0);
                let idx = Self::fnv1a_index(ng, self.vector_dim);
                theme[idx] += tf * idf;
            }
            let norm_sq: f32 = theme.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt().max(1e-8);
            for v in theme.iter_mut() { *v /= norm; }
            self.domain_vectors.insert(domain.id, theme);
        }
    }

    /// 文本 → TF-IDF 固定维度向量（与领域主题向量同 hash 空间）
    fn compute_tfidf_vector(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.vector_dim];
        let ngrams = Self::extract_ngrams(text);
        let total = ngrams.len() as f32;
        if total == 0.0 { return vec; }
        let mut tf_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for ng in ngrams { *tf_map.entry(ng).or_insert(0.0) += 1.0; }
        for (ng, tf) in &tf_map {
            let idf = self.idf.get(ng).copied().unwrap_or(1.0);
            let idx = Self::fnv1a_index(ng, self.vector_dim);
            vec[idx] += (tf / total) * idf;
        }
        let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        for v in vec.iter_mut() { *v /= norm; }
        vec
    }

    /// cosine 相似度（L2 已归一化 → 直接点积，裁剪到 [0, 1]）
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>().max(0.0)
    }

    /// softmax：相似度 → 概率分布（数值稳定版，减 max）
    fn softmax(sims: &[f32]) -> Vec<f32> {
        let max_v = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = Vec::with_capacity(sims.len());
        let mut sum_exp = 0.0f64;
        for s in sims {
            let e = (s - max_v).exp() as f64;
            exps.push(e as f32);
            sum_exp += e;
        }
        let sum_exp = if sum_exp > 0.0 { sum_exp } else { 1.0 };
        exps.iter().map(|e| e / sum_exp as f32).collect()
    }

    /// 提取 char n-gram（n=2,3）
    fn extract_ngrams(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        [2usize, 3].into_iter()
            .flat_map(|n| chars.windows(n).map(|w| w.iter().collect()))
            .collect()
    }

    /// FNV-1a → 桶索引（与 FeatureEncoder 同一 hash 函数，保持桶空间一致）
    fn fnv1a_index(bytes: &str, dim: usize) -> usize {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in bytes.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % dim
    }
}

impl ContentAnalyzer for TFIDFAnalyzer {
    fn analyze(&self, raw_input: &RawInput) -> Result<ContentAnalysis, Box<dyn std::error::Error>> {
        let text = &raw_input.text;

        // Step 1: 输入文本 → TF-IDF 向量 → cosine 相似度 → softmax
        let input_vec = self.compute_tfidf_vector(text);
        let sims: Vec<(u32, f32)> = self.domains.iter().map(|d| {
            let theme = self.domain_vectors.get(&d.id).unwrap();
            (d.id, Self::cosine(&input_vec, theme))
        }).collect();
        let raw_sims: Vec<f32> = sims.iter().map(|(_, s)| *s).collect();
        let softmaxed = Self::softmax(&raw_sims);
        let tfidf_w: Vec<(u32, f32)> = sims.into_iter().zip(softmaxed.iter())
            .map(|((id, _), &s)| (id, s)).collect();

        // Step 2: SyntaxPattern 结构化加分（和 cosine 融合 0.7/0.3）
        let syntax_w = self.syntax_lib.detect(text);
        let sy_map: std::collections::HashMap<u32, f32> = syntax_w.iter().copied().collect();
        let fused: Vec<(u32, f32)> = tfidf_w.iter().map(|(id, tf)| {
            let sy = sy_map.get(id).copied().unwrap_or(0.0);
            (*id, tf * 0.7 + sy * 0.3)
        }).collect();

        // Step 3: MetaRouter.adjust → metadata 辅助
        let adjusted = self.meta_router.adjust(&fused, &raw_input.metadata, raw_input.timestamp);

        // Step 4: 生成 Evidence（TF-IDF + 结构化命中）
        let mut evidence: Vec<Evidence> = Vec::new();
        for (did, sy) in &syntax_w {
            if *sy > 0.0 {
                let name = self.domains.iter().find(|d| d.id == *did)
                    .map(|d| d.name.as_str()).unwrap_or("未知");
                evidence.push(Evidence {
                    domain_id: *did,
                    description: format!("结构化模式命中（领域：{}）", name),
                    contribution: sy * 0.3,
                });
            }
        }

        // Step 5: FeatureEncoder Tensor + PaddingMask
        let feature_tensor = self.feature_encoder.encode(text, &self.domains);
        let mut analysis = ContentAnalysis::new(adjusted, feature_tensor);
        analysis.evidence = evidence;
        if let Some(ref lengths) = raw_input.sample_lengths {
            let max_len = raw_input.max_padded_length
                .unwrap_or_else(|| *lengths.iter().max().unwrap_or(&256));
            analysis = analysis.with_padding_mask(PaddingMask::detect(lengths, max_len));
        }

        Ok(analysis)
    }

    fn domains(&self) -> &[Domain] { &self.domains }
    // TFIDFAnalyzer 没有硬规则（空 Vec，让 HybridAnalyzer 硬规则优先判断）
    fn rule_based_match(&self, _text: &str) -> Vec<(u32, f32)> { Vec::new() }
    fn train_semantic_encoder(&mut self, _training_data: &[crate::traits::analyzer::LabeledContent]) {}
    fn feature_dim(&self) -> usize { self.feature_encoder.effective_output_dim() }
}
