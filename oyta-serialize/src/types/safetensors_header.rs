//! SafetensorsHeader + TensorEntry：safetensors 文件 JSON header 结构
//!
//! safetensors v0 格式：
//!   JSON header 里每个 key 对应一个 TensorEntry
//!   "__metadata__" key 存全局元数据（oyta_version 等）

use serde::{Deserialize, Serialize};

/// safetensors 文件单个 Tensor 的 header entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// dtype：目前只支持 "F32"（oyta-frame 统一 f32 精度）
    pub dtype: String,
    /// shape：如 [128, 64]
    pub shape: Vec<usize>,
    /// data_offsets：[start, end] — 相对 header 结束位置的字节偏移
    pub data_offsets: [usize; 2],
}

/// safetensors 文件 header（完整 JSON）
#[derive(Debug, Clone, Default)]
pub struct SafetensorsHeader {
    /// 所有 tensor entry：key = Parameter.name（如 "domain_1.mlp.hidden.weight"）
    pub tensors: std::collections::HashMap<String, TensorEntry>,
    /// 全局元数据（key "__metadata__"）
    pub metadata: std::collections::HashMap<String, String>,
}

impl SafetensorsHeader {
    /// 构造空 header + 默认 oyta_version 元数据
    pub fn new() -> Self {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("oyta_version".to_string(), "0.1.0".to_string());
        Self { tensors: std::collections::HashMap::new(), metadata }
    }

    /// 序列化到 serde_json::Value
    ///
    /// tensors 先写，__metadata__ 最后追加（safetensors 规范要求）
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (k, v) in &self.tensors {
            map.insert(k.clone(), serde_json::to_value(v).unwrap());
        }
        if !self.metadata.is_empty() {
            let mut md = serde_json::Map::new();
            for (k, v) in &self.metadata {
                md.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            map.insert("__metadata__".to_string(), serde_json::Value::Object(md));
        }
        serde_json::Value::Object(map)
    }

    /// 从 serde_json::Value 反序列化
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("header 不是 JSON object")?;
        let mut tensors = std::collections::HashMap::new();
        let mut metadata = std::collections::HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                if let Some(md_obj) = v.as_object() {
                    for (mk, mv) in md_obj {
                        if let Some(s) = mv.as_str() { metadata.insert(mk.clone(), s.to_string()); }
                    }
                }
            } else {
                let te: TensorEntry = serde_json::from_value(v.clone())
                    .map_err(|e| format!("解析 tensor entry 失败: {}", e))?;
                tensors.insert(k.clone(), te);
            }
        }
        Ok(Self { tensors, metadata })
    }
}
