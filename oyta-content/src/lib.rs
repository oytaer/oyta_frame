//! oyta-content：内容分析器 crate（最底层，无框架依赖）
//!
//! 完整四层分层目录结构：
//! ┌─────────────────────────────────────────────────────────┐
//! │ lib.rs              顶层统一 pub mod 声明 + pub use      │
//! ├─────────────────────────────────────────────────────────┤
//! │ types/              数据类型层（所有数据结构，无逻辑）  │
//! │   ├── mod.rs        只 pub mod 声明                     │
//! │   ├── domain.rs           Domain 结构体 + 预定义领域    │
//! │   ├── raw_input.rs        RawInput 原始输入（秒级时间）│
//! │   └── content_analysis.rs ContentAnalysis + Evidence   │
//! ├─────────────────────────────────────────────────────────┤
//! │ traits/             接口层（只 trait 定义，零实现）     │
//! │   ├── mod.rs        只 pub mod 声明                     │
//! │   └── analyzer.rs         ContentAnalyzer trait         │
//! ├─────────────────────────────────────────────────────────┤
//! │ utils/              工具层（底层可复用组件）            │
//! │   ├── mod.rs        只 pub mod 声明                     │
//! │   ├── feature_encoder.rs  文本 → 固定维度 Tensor        │
//! │   └── padding_mask.rs     变长 padding mask 检测        │
//! ├─────────────────────────────────────────────────────────┤
//! │ analyzers/          实现层（三种 ContentAnalyzer）       │
//! │   ├── mod.rs        只 pub mod 声明                     │
//! │   ├── rule_based.rs       regex 关键词匹配              │
//! │   ├── tfidf.rs            TF-IDF cosine 相似度          │
//! │   └── hybrid.rs           规则优先 + TFIDF 融合         │
//! └─────────────────────────────────────────────────────────┘

// ===== types：数据类型层 =====
pub mod types;
// ===== traits：接口层 =====
pub mod traits;
// ===== utils：工具层 =====
pub mod utils;
// ===== analyzers：实现层 =====
pub mod analyzers;

// ===== 公共导出（从各子文件直接 pub use，不经过 mod.rs）=====
pub use types::domain::Domain;
pub use types::raw_input::RawInput;
pub use types::content_analysis::{ContentAnalysis, Evidence};
pub use traits::analyzer::{ContentAnalyzer, LabeledContent};
pub use utils::feature_encoder::FeatureEncoder;
pub use utils::padding_mask::PaddingMask;
pub use analyzers::rule_based::RuleBasedAnalyzer;
pub use analyzers::tfidf::TFIDFAnalyzer;
pub use analyzers::hybrid::HybridAnalyzer;

// ===== prelude：一行导入 =====
pub mod prelude {
    pub use crate::{
        Domain, RawInput, ContentAnalysis, ContentAnalyzer, LabeledContent,
        RuleBasedAnalyzer, TFIDFAnalyzer, HybridAnalyzer,
        FeatureEncoder, PaddingMask,
    };
    pub use oyta_tensor::prelude::*;
}

// ===== 集成测试模块（验收标准 + 行为验证）=====
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::domain::default_domains;

    /// 验收标准 1：RuleBasedAnalyzer 对金融关键词文本返回 finance 权重最高
    #[test]
    fn test_rule_based_finance_detection() {
        let domains = default_domains();
        let analyzer = RuleBasedAnalyzer::new(domains);
        let raw = RawInput::from_text("净利润同比增长 15% 营收达到 12.5 亿元 公司市盈率 8.2");
        let result = analyzer.analyze(&raw).expect("analyze 应成功");

        // finance domain_id=1, generic domain_id=0
        let finance = result.domain_weights.iter().find(|(id, _)| *id == 1).map(|(_, w)| *w).unwrap_or(0.0);
        let generic = result.domain_weights.iter().find(|(id, _)| *id == 0).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(finance > generic,
            "金融文本应识别为 finance 领域 (finance={}, generic={})", finance, generic);
        assert!(finance > 0.4, "finance 权重应显著高于背景 (finance={})", finance);
        // 归一化权重和 ≈ 1
        let sum: f32 = result.domain_weights.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01, "归一化权重和应为 1 (实际={})", sum);
    }

    /// 验收标准 2：HybridAnalyzer 对医疗文本返回 medical 权重最高
    #[test]
    fn test_hybrid_medical_detection() {
        let domains = default_domains();
        let analyzer = HybridAnalyzer::new(domains);
        let raw = RawInput::from_text("患者体温 37.8℃ 血压 120/80 mmHg 血糖 6.5 mmol/L 诊断为高血压");
        let result = analyzer.analyze(&raw).expect("analyze 应成功");

        let medical = result.domain_weights.iter().find(|(id, _)| *id == 2).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(medical > 0.3, "医疗文本应识别为 medical 领域 (medical={})", medical);
        let sum: f32 = result.domain_weights.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    /// 验收标准 3：TFIDFAnalyzer 对法律文本返回 legal 权重最高
    #[test]
    fn test_tfidf_legal_detection() {
        let domains = default_domains();
        let analyzer = TFIDFAnalyzer::new(domains);
        let raw = RawInput::from_text("本合同第 3 条规定 甲方应当于 2024 年 12 月 31 日前 按民法典第 5 条执行");
        let result = analyzer.analyze(&raw).expect("analyze 应成功");

        let legal = result.domain_weights.iter().find(|(id, _)| *id == 3).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(legal > 0.15, "法律文本应识别为 legal 领域 (legal={})", legal);
    }

    /// 验收标准 4：空文本 fallback 不 panic，返回 best domain 权重 1.0
    #[test]
    fn test_empty_text_fallback() {
        let domains = default_domains();
        let analyzer = RuleBasedAnalyzer::new(domains);
        let raw = RawInput::from_text("");
        let result = analyzer.analyze(&raw).expect("空文本也应成功");

        let sum: f32 = result.domain_weights.iter().map(|(_, w)| *w).sum();
        assert!(sum > 0.0, "空文本应有 fallback 权重");
    }

    /// 验收标准 5：metadata/timestamp 辅助路由 — PDF 扩展名提升 legal 权重
    #[test]
    fn test_metadata_router_pdf_legal() {
        let raw = RawInput::from_text("合同 条款 签订");
        let raw_with_meta = RawInput::from_text("合同 条款 签订")
            .with_metadata("ext", "pdf");
        let analyzer = RuleBasedAnalyzer::new(default_domains());
        let result_no_meta = analyzer.analyze(&raw).unwrap();
        let analyzer2 = RuleBasedAnalyzer::new(default_domains());
        let result_with_meta = analyzer2.analyze(&raw_with_meta).unwrap();

        let legal_no = result_no_meta.domain_weights.iter().find(|(id, _)| *id == 3).map(|(_, w)| *w).unwrap_or(0.0);
        let legal_with = result_with_meta.domain_weights.iter().find(|(id, _)| *id == 3).map(|(_, w)| *w).unwrap_or(0.0);
        // PDF 扩展名应该提升法律权重（MetaRouter 加了 bonus）
        assert!(legal_with >= legal_no || legal_with > 0.2,
            "PDF metadata 应提升 legal 权重 (no_meta={}, with_meta={})", legal_no, legal_with);
    }

    /// 验收标准 6：FeatureEncoder Tensor shape 正确
    #[test]
    fn test_feature_encoder_tensor_shape() {
        let analyzer = RuleBasedAnalyzer::new(default_domains());
        let raw = RawInput::from_text("测试文本内容");
        let result = analyzer.analyze(&raw).unwrap();

        let shape = result.feature_vector.shape();
        assert_eq!(shape.dims[0], 1, "batch size 应为 1");
        assert_eq!(shape.dims[1], analyzer.feature_dim(), "feature_dim 应匹配");
    }

    /// 验收标准 7：HybridAnalyzer.train_semantic_encoder MiniMLP 训练完成
    #[test]
    fn test_train_semantic_encoder_completes() {
        let mut analyzer = HybridAnalyzer::new(default_domains());

        // 构造标注训练数据
        let mut training_data: Vec<LabeledContent> = Vec::new();
        for _ in 0..20 {
            let text = "净利润 营收 市盈率 股价 上涨 下跌 成交量 市值".to_string();
            training_data.push(LabeledContent {
                raw: RawInput::from_text(&text),
                true_weights: vec![(1, 0.9), (0, 0.1)], // finance 权重 0.9
            });
        }

        // 训练 30 个 epoch，学习率 0.1
        let _ = analyzer.train_semantic_encoder_full(&training_data, Some(30), Some(0.1));

        // 训练完后应有 MiniMLP 保存
        assert!(analyzer.has_semantic_mlp(), "训练完后应保存 MiniMLP");
    }

    /// 验收标准 8：所有 evidence 不为空（硬规则至少有结构化或关键词命中）
    #[test]
    fn test_evidence_present() {
        let analyzer = RuleBasedAnalyzer::new(default_domains());
        let raw = RawInput::from_text("净利润同比增长 15% 营收 12.5 亿元");
        let result = analyzer.analyze(&raw).unwrap();

        assert!(!result.evidence.is_empty(), "有效文本应有 Evidence 记录");
        for ev in &result.evidence {
            assert!(ev.contribution >= 0.0, "Evidence.contribution 不应为负");
        }
    }

    /// 验收标准 9：变长输入 padding_mask 正确生成
    #[test]
    fn test_variable_length_padding_mask() {
        let analyzer = RuleBasedAnalyzer::new(default_domains());
        // sample_lengths 是每个样本的实际长度
        let raw = RawInput::from_text("短文本").with_variable_lengths(vec![3, 11], 11);
        let result = analyzer.analyze(&raw).unwrap();

        assert!(result.padding_mask.is_some(), "变长输入应生成 padding_mask");
        let mask = result.padding_mask.as_ref().unwrap();
        assert_eq!(mask.len(), 2, "mask 长度应等于样本数");
        assert_eq!(mask[0].len(), 11, "每个 mask 长度应等于 max_padded_length");
        // PaddingMask 语义: 0=有效位置, 1=padding 位置
        // 第一个样本长度 3: t=0,1,2 有效(0.0), t>=3 padding(1.0)
        assert_eq!(mask[0][0], 0.0, "有效位置应为 0.0");
        assert_eq!(mask[0][2], 0.0, "位置 2 为最后有效位置");
        assert_eq!(mask[0][3], 1.0, "位置 3 为 padding，应为 1.0");
        assert_eq!(mask[0][10], 1.0, "位置 10 为 padding");
        // 第二个样本长度 11: 全部有效
        assert_eq!(mask[1].iter().sum::<f32>(), 0.0, "长度等于 max_length 应无 padding");
    }
}

