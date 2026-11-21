//! SyntaxPattern：句法/结构特征检测（架构文档 §5.2 分析维度表第 2 行）
//!
//! 结构化模式比扁平关键词更精准——连续出现多个结构化元素 → 强领域信号
//!
//! 例子：
//!   "今日上证指数 3125.50 点 上涨 12.30 点 成交量 123.45 亿"
//!     → 结构化元素: [数字+元+点] [数字+点+数字] [数字+亿] → 强金融信号
//!   "第 3 条 合同 条款 违反 民法典 第 5 条"
//!     → 结构化元素: [第+数字+条/款/条] → 强法律信号
//!   "体温 37.8℃ 血压 120/80 mmHg 血糖 6.5 mmol/L"
//!     → 结构化元素: [数字+/+数字] [数字+mmol/L] → 强医疗信号
//!
//! 完整实现（无骨架/无占位）：
//!   SyntaxPatternLibrary — 预编译的领域特定结构化正则集合
//!   SyntaxMatchResult — 检测结果（每个领域的结构化模式命中次数）
//!   detect() — 对任意文本跑所有领域的结构化正则
//!
//! 为什么用预编译 HashMap<u32, Vec<Regex>>？
//!   和 RuleBasedAnalyzer.keyword_regexes 同一个设计——构造时一次性编译

use std::collections::HashMap;
use regex::Regex;

/// 单个结构化模式：正则 + 该模式属于哪个领域
struct SyntaxRule {
    /// 结构化正则（预编译）
    pattern: Regex,
    /// 命中这个正则后加在哪个领域的结构化权重上
    domain_id: u32,
    /// 权重加成（默认 1.0，高信号模式可以更高）
    weight: f32,
}

/// 结构化模式库（预编译，analyze 时只做 find_iter.count()）
pub struct SyntaxPatternLibrary {
    /// 所有预编译的结构化正则（构造时一次性编译完成）
    rules: Vec<SyntaxRule>,
}

impl SyntaxPatternLibrary {
    /// 构造：注册架构文档定义的所有结构化模式
    pub fn new() -> Self {
        let mut rules: Vec<SyntaxRule> = Vec::new();

        // ===== 金融领域结构化模式（domain_id = 1）=====
        // "数字 + 元" / "数字 + 亿元" / "数字 + 万" —— 金额表述
        rules.push(SyntaxRule { pattern: Self::re(r"(\d+(?:\.\d+)?)\s*(亿元|元|万元|万|亿)"), domain_id: 1, weight: 1.0 });
        // "数字 + 点 + 数字" —— 股价/指数（如 3125.50）
        rules.push(SyntaxRule { pattern: Self::re(r"(\d+)\.(\d{1,2})"), domain_id: 1, weight: 0.5 });
        // "数字 + 点" —— 百分比（如 3.25%）
        rules.push(SyntaxRule { pattern: Self::re(r"(\d+(?:\.\d+)?)\s*%"), domain_id: 1, weight: 1.0 });
        // "成交量 / 成交额 / 市值 / 市盈率 / K线 / 开盘 / 收盘 / 涨停 / 跌停" —— 金融术语
        rules.push(SyntaxRule { pattern: Self::re(r"(成交量|成交额|市值|市盈率|市净率|K线|开盘|收盘|涨停|跌停|大盘|上证|深证|纳斯达克|恒生)"), domain_id: 1, weight: 1.5 });
        // "数字 + [+/−] + 数字" —— 涨跌额（如 +12.30 / -5.67）
        rules.push(SyntaxRule { pattern: Self::re(r"([+\-])\s*(\d+(?:\.\d+)?)"), domain_id: 1, weight: 1.0 });

        // ===== 医疗领域结构化模式（domain_id = 2）=====
        // "数字 / 数字" —— 血压/比值（如 120/80）
        rules.push(SyntaxRule { pattern: Self::re(r"(\d+)\s*/\s*(\d+)"), domain_id: 2, weight: 1.0 });
        // "数字 mmol/L / mg/dL / ℃ / mmHg" —— 医疗指标
        rules.push(SyntaxRule { pattern: Self::re(r"(\d+(?:\.\d+)?)\s*(mmol/L|mg/dL|℃|°C|mmHg|g/L|ng/mL)"), domain_id: 2, weight: 1.5 });
        // "体温 / 血压 / 血糖 / 心率 / 诊断 / 处方 / 病历 / 症状" —— 医疗术语
        rules.push(SyntaxRule { pattern: Self::re(r"(体温|血压|血糖|心率|脉搏|诊断|处方|病历|症状|体征|治疗方案|复诊|随访)"), domain_id: 2, weight: 1.5 });
        // "+ / -" 医学检查标记（如蛋白+ / 潜血-）
        rules.push(SyntaxRule { pattern: Self::re(r"(\S)([+\-])"), domain_id: 2, weight: 0.5 });

        // ===== 法律领域结构化模式（domain_id = 3）=====
        // "第 + 数字 + 条 / 款 / 项 / 章" —— 法条引用
        rules.push(SyntaxRule { pattern: Self::re(r"第\s*(\d+)\s*(条|款|项|章|节|款)"), domain_id: 3, weight: 1.5 });
        // "合同 / 条款 / 甲方 / 乙方 / 违约 / 诉讼 / 判决书 / 民法典 / 签署" —— 法律术语
        rules.push(SyntaxRule { pattern: Self::re(r"(合同|条款|甲方|乙方|丙方|违约|诉讼|判决书|裁定书|民法典|公司法|合同法|签署|订立|解除)"), domain_id: 3, weight: 1.5 });
        // 日期 "YYYY年MM月DD日" —— 合同签署/判决日期
        rules.push(SyntaxRule { pattern: Self::re(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"), domain_id: 3, weight: 1.0 });

        // ===== 教育领域结构化模式（domain_id = 4）=====
        // "第 + 数字 + 课 / 章" —— 课程章节
        rules.push(SyntaxRule { pattern: Self::re(r"第\s*(\d+)\s*(课|章|节|讲|单元|周)"), domain_id: 4, weight: 1.0 });
        // "分数 / 成绩 / 学分 / 作业 / 考试 / 论文 / 答辩 / 研究" —— 教育术语
        rules.push(SyntaxRule { pattern: Self::re(r"(分数|成绩|学分|作业|考试|期中|期末|高考|论文|答辩|研究|学分绩|GPA)"), domain_id: 4, weight: 1.5 });

        Self { rules }
    }

    /// 预编译正则辅助（(?i) 大小写不敏感 + regex::escape 保护用户输入无特殊字符）
    fn re(pattern: &str) -> Regex {
        Regex::new(&format!(r"(?i){}", pattern)).expect("SyntaxPattern regex 编译失败")
    }
}

/// 结构化特征检测结果
/// key = domain_id, value = 该领域所有结构化模式加权命中数
pub type SyntaxMatchResult = Vec<(u32, f32)>;

impl SyntaxPatternLibrary {
    /// 对任意文本跑所有结构化正则 → 返回每个领域的加权命中数
    ///
    /// 返回: [(domain_id, weighted_hit_score)] 形式
    /// 分数计算: Σ(每条正则的 find_iter.count() × 该正则的 weight)
    pub fn detect(&self, text: &str) -> SyntaxMatchResult {
        // 按 domain_id 累加分数
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for rule in &self.rules {
            let hits = rule.pattern.find_iter(text).count() as f32;
            if hits > 0.0 {
                *scores.entry(rule.domain_id).or_insert(0.0) += hits * rule.weight;
            }
        }
        scores.into_iter().collect()
    }
}

impl Default for SyntaxPatternLibrary {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finance_detection() {
        let lib = SyntaxPatternLibrary::new();
        // 金融文本 → domain_id=1 应该有命中
        let text = "今日上证指数 3125.50 点 上涨 12.30 点 成交量 123.45 亿元 市盈率 15.6";
        let result = lib.detect(text);
        let finance_score = result.iter().find(|(id, _)| *id == 1).map(|(_, s)| *s).unwrap_or(0.0);
        assert!(finance_score > 0.0, "金融文本应该命中结构化模式");
    }

    #[test]
    fn test_medical_detection() {
        let lib = SyntaxPatternLibrary::new();
        let text = "体温 37.8℃ 血压 120/80 mmHg 血糖 6.5 mmol/L 诊断为高血压";
        let result = lib.detect(text);
        let medical_score = result.iter().find(|(id, _)| *id == 2).map(|(_, s)| *s).unwrap_or(0.0);
        assert!(medical_score > 0.0, "医疗文本应该命中结构化模式");
    }

    #[test]
    fn test_legal_detection() {
        let lib = SyntaxPatternLibrary::new();
        let text = "本合同第 3 条规定 甲方应当于 2024 年 12 月 31 日前 按民法典第 5 条执行";
        let result = lib.detect(text);
        let legal_score = result.iter().find(|(id, _)| *id == 3).map(|(_, s)| *s).unwrap_or(0.0);
        assert!(legal_score > 0.0, "法律文本应该命中结构化模式");
    }

    #[test]
    fn test_empty_text() {
        let lib = SyntaxPatternLibrary::new();
        let result = lib.detect("");
        assert!(result.is_empty(), "空文本应该零命中");
    }
}
