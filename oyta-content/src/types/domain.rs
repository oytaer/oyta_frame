//! Domain：领域注册表
//!
//! 每个 Domain 代表一个独立的内容领域（如金融 / 医疗 / 法律 / 教育）
//! 每个领域有自己的关键词词典（供 RuleBasedAnalyzer 硬规则匹配）
//!
//! 约束：
//!   - id 从 0 开始，稳定不变（DomainRouter 用 id 做 Vec 索引）
//!   - keywords 大小写不敏感匹配（regex 编译时加 (?i) flag）

use serde::{Deserialize, Serialize};

/// 领域定义结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    /// 领域唯一 ID（从 0 开始，稳定不变，Router 用它做 mask 索引）
    pub id: u32,
    /// 领域英文名称（snake_case，如 "finance" / "medical"）
    pub name: String,
    /// 领域中文描述（人类可读，如 "金融领域"）
    pub description: String,
    /// 关键词词典（RuleBasedAnalyzer 在输入文本里搜索这些关键词，大小写不敏感）
    pub keywords: Vec<String>,
    /// 该领域激活的最低权重阈值（DomainRouter 判断领域是否激活用）
    pub weight_threshold: f32,
    /// 该领域在 FeatureEncoder 特征向量里的偏移量（保留字段，预留给高级特征编码）
    pub feature_offset: usize,
}

impl Domain {
    /// 最简构造（只填 id + name，其他用默认值）
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: String::new(),
            keywords: Vec::new(),
            weight_threshold: 0.1,     // 默认 0.1（低于这个权重认为没命中）
            feature_offset: 0,        // 默认 0（预留给高级 FeatureEncoder）
        }
    }

    /// 链式添加关键词词典
    pub fn with_keywords(mut self, keywords: Vec<impl Into<String>>) -> Self {
        self.keywords = keywords.into_iter().map(|k| k.into()).collect();
        self
    }

    /// 链式设置激活阈值
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.weight_threshold = threshold;
        self
    }

    /// 链式设置中文描述
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// 预定义 5 个标准领域（启动时默认注册，用户可以覆盖 / 扩展）
pub fn default_domains() -> Vec<Domain> {
    vec![
        Domain::new(0, "generic")
            .with_description("通用领域（兜底，匹配不到其他领域时激活）")
            .with_keywords(vec![
                "the", "is", "are", "was", "were", "and", "or", "to", "of", "in",
            ])
            .with_threshold(0.05),  // 通用领域阈值低，让它容易被激活（兜底用）

        Domain::new(1, "finance")
            .with_description("金融领域（股票、基金、财报）")
            .with_keywords(vec![
                "利润", "营收", "净利润", "净利润率", "股价", "K线", "成交量",
                "市盈率", "股息", "年报", "季报", "股东", "分红", "财报",
            ])
            .with_threshold(0.1),

        Domain::new(2, "medical")
            .with_description("医疗领域（诊断、处方、病历）")
            .with_keywords(vec![
                "患者", "症状", "诊断", "处方", "病历", "血压", "血糖",
                "体温", "治疗方案", "药物", "疗程", "病理",
            ])
            .with_threshold(0.1),

        Domain::new(3, "legal")
            .with_description("法律领域（合同、条款、诉讼）")
            .with_keywords(vec![
                "合同", "条款", "甲方", "乙方", "违约", "诉讼", "判决书",
                "民法典", "公司法", "签署", "法律效力", "诉讼时效",
            ])
            .with_threshold(0.1),

        Domain::new(4, "education")
            .with_description("教育领域（课程、作业、考试）")
            .with_keywords(vec![
                "课程", "作业", "考试", "成绩", "学生", "教师", "论文",
                "课题", "学分", "毕业", "学术",
            ])
            .with_threshold(0.1),
    ]
}
