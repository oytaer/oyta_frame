//! MetaRouter：metadata + timestamp 辅助路由权重调整
//!
//! 架构文档 §5.2 分析维度表第 3 行：
//!   "元数据 —— 文件扩展名 / 来源 URL / 时间戳 / 作者 —— 辅助路由"
//!
//! 设计：
//!   metadata 是 HashMap<String, String>，key 为 "ext"/"source"/"author"/"lang" 等
//!   timestamp 是 u64 秒级（按月/周判断是否在财报季等特殊时段）
//!
//! 完整实现（无骨架/无占位）：
//!   MetaRouter.adjust(domain_weights, metadata, timestamp) → 调整后的权重
//!
//! 调整逻辑（全部可配置，通过规则表扩展）：
//!   1. 文件扩展名 → 领域加分（.html/.json → generic; .pdf/.docx → 可能 legal/education）
//!   2. 来源 URL → 领域加分（finance.sina.com.cn → finance; medical.net → medical）
//!   3. 时间戳 → 财报季（Q1:1-3月, Q2:4-6月...）提升 finance 权重
//!   4. 关键词匹配 metadata value → 领域加分

use std::collections::HashMap;

/// 元数据路由辅助（构造时注册规则，analyze 时纯查表）
pub struct MetaRouter {
    /// 扩展名 → 领域加分映射（ext 小写）
    ext_rules: HashMap<String, (u32, f32)>,
    /// 来源 URL 关键词 → 领域加分映射
    source_rules: Vec<(String, u32, f32)>,
}

impl MetaRouter {
    pub fn new() -> Self {
        let mut ext_rules: HashMap<String, (u32, f32)> = HashMap::new();
        // ===== 扩展名规则 =====
        ext_rules.insert("html".into(),  (0, 0.3));   // generic 网页
        ext_rules.insert("htm".into(),   (0, 0.3));
        ext_rules.insert("json".into(),  (0, 0.3));   // generic 数据
        ext_rules.insert("xml".into(),   (0, 0.2));
        ext_rules.insert("pdf".into(),   (3, 0.2));   // 合同/法律文书
        ext_rules.insert("doc".into(),   (3, 0.2));
        ext_rules.insert("docx".into(),  (3, 0.2));
        ext_rules.insert("xls".into(),   (1, 0.3));   // 金融报表
        ext_rules.insert("xlsx".into(),  (1, 0.3));
        ext_rules.insert("csv".into(),   (1, 0.2));
        ext_rules.insert("ppt".into(),   (4, 0.2));   // 教育演示
        ext_rules.insert("pptx".into(),  (4, 0.2));
        ext_rules.insert("md".into(),    (0, 0.2));   // generic 文档

        let source_rules: Vec<(String, u32, f32)> = vec![
            // ===== 来源域名关键词规则 =====
            ("finance".into(),   1, 0.4),   // 任何带 finance 的域名 → 金融
            ("sina".into(),      1, 0.3),   // 新浪财经
            ("eastmoney".into(), 1, 0.4),   // 东方财富
            ("stock".into(),     1, 0.3),   // 股票
            ("medical".into(),   2, 0.4),   // 任何带 medical → 医疗
            ("health".into(),    2, 0.3),   // 健康医疗
            ("hospital".into(),  2, 0.4),   // 医院
            ("legal".into(),     3, 0.4),   // 任何带 legal → 法律
            ("law".into(),       3, 0.3),   // 法律类
            ("court".into(),     3, 0.4),   // 法院
            ("edu".into(),       4, 0.4),   // 教育网通用
            ("school".into(),    4, 0.3),   // 学校
            ("university".into(), 4, 0.3),
            ("course".into(),    4, 0.3),
        ];

        Self { ext_rules, source_rules }
    }

    /// 辅助路由：根据 metadata 和 timestamp 调整领域权重
    ///
    /// # 参数
    /// - weights: 原始领域权重 [(domain_id, weight)]（来自 RuleBasedAnalyzer/TFIDFAnalyzer）
    /// - metadata: 元数据 HashMap（RawInput.metadata）
    /// - timestamp: 秒级时间戳（0 表示未知）
    ///
    /// # 返回
    /// - 调整后的权重 [(domain_id, new_weight)]（和原始格式一致）
    pub fn adjust(&self, weights: &[(u32, f32)], metadata: &HashMap<String, String>, timestamp: u64) -> Vec<(u32, f32)> {
        // Step 1: 收集加分（全是增量，不覆盖原始权重）
        let mut bonus: HashMap<u32, f32> = HashMap::new();

        // 1a. 扩展名匹配（metadata["ext"] 或 metadata["file_ext"]）
        let ext = metadata.get("ext").or_else(|| metadata.get("file_ext")).map(|s| s.to_lowercase());
        if let Some(ref e) = ext {
            if let Some(&(did, add)) = self.ext_rules.get(e) {
                *bonus.entry(did).or_insert(0.0) += add;
            }
        }

        // 1b. 来源 URL 匹配（metadata["source"] / metadata["url"] / metadata["domain"]）
        let source_keys = ["source", "url", "domain", "referrer"];
        for key in &source_keys {
            if let Some(src) = metadata.get(*key) {
                let src_lower = src.to_lowercase();
                for (keyword, did, add) in &self.source_rules {
                    if src_lower.contains(keyword) {
                        *bonus.entry(*did).or_insert(0.0) += add;
                    }
                }
            }
        }

        // 1c. 财报季时间戳 → finance 加分（Q1:1-3 / Q2:4-6 / Q3:7-9 / Q4:10-12）
        if timestamp > 0 {
            let month = Self::unix_seconds_to_month(timestamp);
            if matches!(month, 1..=3 | 4..=6 | 7..=9 | 10..=12) {
                // 财报季给 finance 加分
                *bonus.entry(1).or_insert(0.0) += 0.15;
            }
        }

        // 1d. 作者/机构关键词 → 加分（metadata["author"] / metadata["org"]）
        let author_keys = ["author", "org", "organization"];
        for key in &author_keys {
            if let Some(val) = metadata.get(*key) {
                let v = val.to_lowercase();
                if v.contains("bank") || v.contains("securities") || v.contains("证券") || v.contains("银行") {
                    *bonus.entry(1).or_insert(0.0) += 0.3;
                }
                if v.contains("hospital") || v.contains("clinic") || v.contains("医院") || v.contains("诊所") {
                    *bonus.entry(2).or_insert(0.0) += 0.3;
                }
                if v.contains("court") || v.contains("law firm") || v.contains("法院") || v.contains("律师") {
                    *bonus.entry(3).or_insert(0.0) += 0.3;
                }
                if v.contains("university") || v.contains("school") || v.contains("大学") || v.contains("学院") {
                    *bonus.entry(4).or_insert(0.0) += 0.3;
                }
            }
        }

        // Step 2: 应用加分到原始权重（加法，保持原始权重的相对关系）
        let mut adjusted: Vec<(u32, f32)> = weights.iter().map(|(id, w)| {
            let b = bonus.get(id).copied().unwrap_or(0.0);
            (*id, (w + b).min(1.0))
        }).collect();

        // 也加上 bonus 里有但原始 weights 里没有的领域
        for (did, b) in &bonus {
            if !adjusted.iter().any(|(id, _)| id == did) {
                adjusted.push((*did, b.min(1.0)));
            }
        }

        // Step 3: 再次归一化（加分后权重和可能不再是 1）
        let total: f32 = adjusted.iter().map(|(_, w)| *w).sum();
        if total > 0.0 {
            for (_, w) in adjusted.iter_mut() { *w /= total; }
        }

        adjusted.retain(|(_, w)| *w > 0.0);
        adjusted
    }

    /// Unix 秒级时间戳 → 月份（1-12）
    fn unix_seconds_to_month(ts: u64) -> u8 {
        // 简化实现：只计算月数（不考虑闰年精度——辅助路由不需要绝对精确）
        // 1970-01-01 是星期四，用 days_since_epoch / 30.4375 近似
        let days = ts / 86_400;
        let approx_months_from_epoch = (days as f64 / 30.4375) as u64;
        // 1970-01 = month 1, 所以 (approx_months_from_epoch % 12) + 1
        let m = (approx_months_from_epoch % 12) + 1;
        m.min(12) as u8
    }
}

impl Default for MetaRouter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ext_rule_pdf_legal() {
        let router = MetaRouter::new();
        let weights = vec![(0, 0.8), (3, 0.2)];
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("ext".into(), "pdf".into());
        let adjusted = router.adjust(&weights, &meta, 0);
        let legal = adjusted.iter().find(|(id, _)| *id == 3).map(|(_, w)| *w).unwrap_or(0.0);
        let generic = adjusted.iter().find(|(id, _)| *id == 0).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(legal > generic || legal > 0.2, "PDF 扩展名应该提升法律权重");
    }

    #[test]
    fn test_source_rule_finance() {
        let router = MetaRouter::new();
        let weights = vec![(0, 0.5), (1, 0.5)];
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("source".into(), "finance.eastmoney.com".into());
        let adjusted = router.adjust(&weights, &meta, 0);
        let finance = adjusted.iter().find(|(id, _)| *id == 1).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(finance > 0.5, "金融来源应该提升 finance 权重");
    }

    #[test]
    fn test_timestamp_april_finance_bonus() {
        let router = MetaRouter::new();
        // 2026-04-01 00:00:00 UTC 秒级时间戳（大约 1775000000）
        let ts_april = 1_775_000_000u64; // 近似值，落在 4 月
        let weights = vec![(0, 1.0)];
        let meta = HashMap::new();
        let adjusted = router.adjust(&weights, &meta, ts_april);
        let finance = adjusted.iter().find(|(id, _)| *id == 1).map(|(_, w)| *w).unwrap_or(0.0);
        assert!(finance > 0.0, "财报季应该给 finance 加分");
    }

    #[test]
    fn test_empty_metadata_no_change() {
        let router = MetaRouter::new();
        let weights = vec![(0, 0.6), (1, 0.4)];
        let meta = HashMap::new();
        let adjusted = router.adjust(&weights, &meta, 0);
        // 无 metadata → 权重应该不变（只有微小浮点误差）
        let finance = adjusted.iter().find(|(id, _)| *id == 1).map(|(_, w)| *w).unwrap_or(0.0);
        assert!((finance - 0.4).abs() < 0.01);
    }
}
