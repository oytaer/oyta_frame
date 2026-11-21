//! RawInput：原始输入包装
//!
//! ContentAnalyzer.analyze() 的统一输入类型
//! 支持：
//!   - 纯文本（text 字段必填）
//!   - 可选结构化数据（structured: HashMap，如 CSV 行 / JSON object）
//!   - 可选变长输入（sample_lengths + max_padded_length → 供 PaddingMask 检测 padding）
//!   - 秒级时间戳（禁止毫秒级，全局约定）

use std::collections::HashMap;

/// 原始输入（ContentAnalyzer 统一输入）
#[derive(Debug, Clone)]
pub struct RawInput {
    /// 主内容（文本格式，必填）
    pub text: String,
    /// 可选结构化数据（如 CSV 行 / JSON object 的 key-value）
    pub structured: Option<HashMap<String, String>>,
    /// 元数据（文件扩展名 / 作者 / 来源 URL 等，通用 k-v）
    pub metadata: HashMap<String, String>,
    /// 秒级时间戳（禁止毫秒级，全局约定）
    pub timestamp: u64,
    /// 变长输入：每个样本的实际长度（不含 padding）
    /// 例：batch 3 样本实际长度 = [28, 31, 25]
    /// ContentAnalyzer 检测到这个字段后会生成 padding_mask
    pub sample_lengths: Option<Vec<usize>>,
    /// 变长输入统一 padding 到的长度（max(sample_lengths)）
    pub max_padded_length: Option<usize>,
}

impl RawInput {
    /// 最简构造：纯文本输入
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            structured: None,
            metadata: HashMap::new(),
            timestamp: 0,           // 0 表示未设置，由上层补秒级时间戳
            sample_lengths: None,   // 定长输入（无 padding）
            max_padded_length: None,
        }
    }

    /// 链式设置秒级时间戳
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// 链式添加元数据字段
    pub fn with_metadata(mut self, key: &str, value: impl Into<String>) -> Self {
        self.metadata.insert(key.to_string(), value.into());
        self
    }

    /// 链式设置变长输入（供 PaddingMask 检测 padding）
    pub fn with_variable_lengths(mut self, lengths: Vec<usize>, max_len: usize) -> Self {
        self.sample_lengths = Some(lengths);
        self.max_padded_length = Some(max_len);
        self
    }

    /// 链式设置结构化数据
    pub fn with_structured(mut self, structured: HashMap<String, String>) -> Self {
        self.structured = Some(structured);
        self
    }

    /// 是否为变长输入（有 sample_lengths = true）
    pub fn is_variable_length(&self) -> bool {
        self.sample_lengths.is_some()
    }
}
