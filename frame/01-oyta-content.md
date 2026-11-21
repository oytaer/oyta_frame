# oyta-content 开发文档（Phase 1 · 完整实现版）

> crate：`oyta-content/` | **唯一底层依赖：`oyta-tensor = "0.1.1"`** | 无其他框架层依赖
> 职责：输入文本 → 输出领域权重 + 固定维度 feature_vector Tensor + 可训练语义编码器
>
> **四层分层目录结构**（详见 00-overview.md §八）：
> ```
> oyta-content/src/
> ├── lib.rs           —— 顶层统一 pub mod 声明 + pub use + prelude
> ├── types/           —— 数据类型层（纯结构体：Domain / RawInput / ContentAnalysis）
> ├── traits/          —— 接口层（只 trait：ContentAnalyzer）
> ├── utils/           —— 工具层（FeatureEncoder / PaddingMask / MiniMLP / SyntaxPattern / MetaRouter）
> └── analyzers/       —— 实现层（RuleBasedAnalyzer / TFIDFAnalyzer / HybridAnalyzer）
> ```

---

# 一、Cargo.toml（直接复制粘贴）

```toml
[package]
name = "oyta-content"
version = "0.1.0"
edition = "2024"
# 禁止 rust-version（你的项目 edition=2024 但没锁定 rust 版本）

[dependencies]
# ===== 底层算子引擎（registry 版本，统一入口）=====
oyta-tensor = "0.1.1"

# ===== 框架层内部直接依赖 =====
serde       = { version = "1.0.229", features = ["derive"] }  # Domain 序列化
serde_json  = "1.0.152"
regex       = "1.11.1"    # RuleBasedAnalyzer 预编译关键词 + SyntaxPattern 结构化模式
rayon       = "1.12.0"    # 预留并行（Phase 2 批量 analyze）
thiserror   = "2.0.16"    # 错误类型派生
anyhow      = "1.0.93"    # HybridAnalyzer.train_semantic_encoder 用
```

---

# 二、完整目录结构

```
oyta-content/src/
├── lib.rs                              ← 顶层入口，只做 pub mod 声明 + pub use + prelude
│
├── types/                              ← 数据类型层（纯结构体定义，零业务逻辑）
│   ├── mod.rs                          ← 只 pub mod xxx;（零 pub use，零实现）
│   ├── domain.rs                       ← Domain + default_domains() 5 个预定义领域
│   ├── raw_input.rs                    ← RawInput（秒级 timestamp + metadata HashMap + 变长）
│   └── content_analysis.rs             ← ContentAnalysis + Evidence
│
├── traits/                             ← 接口层（只 trait 定义，零实现）
│   ├── mod.rs                          ← 只 pub mod xxx;（零 pub use，零实现）
│   └── analyzer.rs                     ← ContentAnalyzer trait + LabeledContent
│
├── utils/                               ← 工具层（底层可复用组件，被 analyzers 层依赖）
│   ├── mod.rs                          ← 只 pub mod xxx;（零 pub use，零实现）
│   ├── feature_encoder.rs              ← FNV-1a hash n-gram → Tensor（可选 MiniMLP 投影层）
│   ├── padding_mask.rs                 ← 变长 padding mask 检测（含单元测试）
│   ├── mini_mlp.rs                     ← 可训练轻量 MLP（Linear→GELU→Linear，架构 §5.2 第 5 行）
│   ├── syntax_pattern.rs               ← 句法/结构特征检测（架构 §5.2 第 2 行）
│   └── meta_router.rs                  ← metadata + timestamp 辅助路由（架构 §5.2 第 3 行）
│
└── analyzers/                          ← 实现层（三种 ContentAnalyzer 完整实现）
    ├── mod.rs                          ← 只 pub mod xxx;（零 pub use，零实现）
    ├── rule_based.rs                   ← RuleBasedAnalyzer（regex 关键词 + SyntaxPattern + MetaRouter 集成）
    ├── tfidf.rs                        ← TFIDFAnalyzer（TF-IDF cosine 相似度 + SyntaxPattern + MetaRouter 集成）
    └── hybrid.rs                       ← HybridAnalyzer（规则优先 + TFIDF 融合 + train_semantic_encoder 完整监督训练）
```

---

# 三、lib.rs（顶层入口，只做模块声明和导出）

```rust
//! oyta-content：内容分析器 crate（Phase 1，最底层，无框架依赖）

// ===== 四层分层：types → traits → utils → analyzers =====
pub mod types;
pub mod traits;
pub mod utils;
pub mod analyzers;

// ===== 公共导出（从子文件直接 pub use，不经过 mod.rs）=====
// types 层
pub use types::domain::Domain;
pub use types::raw_input::RawInput;
pub use types::content_analysis::{ContentAnalysis, Evidence};
// traits 层
pub use traits::analyzer::{ContentAnalyzer, LabeledContent};
// utils 层
pub use utils::feature_encoder::FeatureEncoder;
pub use utils::padding_mask::PaddingMask;
pub use utils::mini_mlp::MiniMLP;
pub use utils::syntax_pattern::SyntaxPatternLibrary;
pub use utils::meta_router::MetaRouter;
// analyzers 层
pub use analyzers::rule_based::RuleBasedAnalyzer;
pub use analyzers::tfidf::TFIDFAnalyzer;
pub use analyzers::hybrid::HybridAnalyzer;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{
        Domain, RawInput, ContentAnalysis, Evidence,
        ContentAnalyzer, LabeledContent,
        FeatureEncoder, PaddingMask, MiniMLP, SyntaxPatternLibrary, MetaRouter,
        RuleBasedAnalyzer, TFIDFAnalyzer, HybridAnalyzer,
    };
    pub use oyta_tensor::prelude::*;
}
```

---

# 四、mod.rs 规范（四层都一样）

> 所有 4 个 mod.rs（types/mod.rs, traits/mod.rs, utils/mod.rs, analyzers/mod.rs）**只允许写：**

```rust
//! xxx 层说明（一行中文注释）

pub mod xxx1;
pub mod xxx2;
pub mod xxx3;
```

> **禁止在 mod.rs 里写：**
> - ❌ `pub use xxx::Xxx;`（导出统一走 lib.rs）
> - ❌ 任何函数/结构体/trait 实现
> - ❌ `use` 语句

---

# 五、types 层（数据类型层）

## 5.1 src/types/domain.rs

```rust
//! Domain：领域注册表
use serde::{Deserialize, Serialize};

/// 领域定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    pub id: u32,                     // 唯一领域 ID（domain_mask 索引用）
    pub name: String,                // "finance" / "medical" / "legal"
    pub description: String,         // "金融领域"
    pub keywords: Vec<String>,       // RuleBasedAnalyzer 关键词词典
    pub weight_threshold: f32,       // RoutingStrategy::Threshold 用的最低激活阈值
    pub feature_offset: usize,       // FeatureEncoder 内部偏移量（预留扩展）
}

impl Domain {
    pub fn new(id: u32, name: impl Into<String>) -> Self { /* 标准构造 */ }
    pub fn with_keywords(mut self, ks: Vec<impl Into<String>>) -> Self { /* 链式设关键词 */ }
    pub fn with_threshold(mut self, t: f32) -> Self { /* 链式设阈值 */ }
}

/// 预定义领域（5 个，启动时默认值）
pub fn default_domains() -> Vec<Domain> {
    vec![
        Domain::new(0, "generic").with_keywords(vec!["the","is","are","and","or","to","of"]).with_threshold(0.05),
        Domain::new(1, "finance").with_keywords(vec!["利润","营收","净利润","股价","K线","成交量","市盈率","股息","年报","季报"]).with_threshold(0.1),
        Domain::new(2, "medical").with_keywords(vec!["患者","症状","诊断","处方","病历","血压","血糖","体温","治疗方案"]).with_threshold(0.1),
        Domain::new(3, "legal").with_keywords(vec!["合同","条款","甲方","乙方","违约","诉讼","判决书","民法典","签署"]).with_threshold(0.1),
        Domain::new(4, "education").with_keywords(vec!["课程","作业","考试","成绩","学生","教师","论文","学分"]).with_threshold(0.1),
    ]
}
```

## 5.2 src/types/raw_input.rs

```rust
//! RawInput：原始输入（秒级时间戳 + metadata HashMap + 变长）
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RawInput {
    pub text: String,
    pub structured: Option<HashMap<String, String>>,       // 可选结构化数据
    pub metadata: HashMap<String, String>,                // MetaRouter 辅助路由用
    pub timestamp: u64,                                   // 秒级时间戳（禁止毫秒，架构硬性规定）
    pub sample_lengths: Option<Vec<usize>>,               // 变长检测用（PaddingMask 用）
    pub max_padded_length: Option<usize>,
}

impl RawInput {
    pub fn from_text(text: impl Into<String>) -> Self { /* 默认构造 */ }
    pub fn with_timestamp(mut self, ts: u64) -> Self { /* 链式设置秒级时间戳 */ }
    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self { /* 链式加 metadata */ }
    pub fn with_variable_lengths(mut self, lengths: Vec<usize>, max_len: usize) -> Self { /* 变长模式 */ }
}
```

## 5.3 src/types/content_analysis.rs

```rust
//! ContentAnalysis：ContentAnalyzer 标准输出
use oyta_tensor::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    pub domain_id: u32,        // 哪个领域产生的证据
    pub description: String,   // 可读描述（"关键词命中" / "结构化模式命中"）
    pub contribution: f32,      // 对最终权重的贡献大小
}

pub struct ContentAnalysis {
    pub domain_weights: Vec<(u32, f32)>,     // 归一化后各领域软权重
    pub feature_vector: Tensor<f32, Cpu>,    // 固定维度特征向量（所有专家统一输入）
    pub evidence: Vec<Evidence>,             // 命中证据（可解释性）
    pub padding_mask: Option<Vec<Vec<f32>>>, // 变长输入 mask（None=定长）
}

impl ContentAnalysis {
    pub fn new(weights: Vec<(u32, f32)>, fv: Tensor<f32, Cpu>) -> Self { /* 基础构造 */ }
    pub fn with_padding_mask(mut self, mask: Vec<Vec<f32>>) -> Self { /* 链式加 mask */ }
}
```

---

# 六、traits 层（接口层）

## 6.1 src/traits/analyzer.rs

```rust
//! ContentAnalyzer trait：所有分析器基类
//!
//! 五个方法的职责：
//!   analyze()            —— 主入口：文本 → ContentAnalysis（权重 + Tensor + Evidence + mask）
//!   domains()            —— 返回所有注册领域（FeatureEncoder 内部可能遍历）
//!   rule_based_match()   —— 返回纯关键词匹配结果（HybridAnalyzer 硬规则判断用）
//!   train_semantic_encoder() —— 可选：轻量语义编码监督训练（架构 §5.2 第 5 行）
//!   feature_dim()        —— FeatureEncoder 输出维度

use crate::types::domain::Domain;
use crate::types::raw_input::RawInput;
use crate::types::content_analysis::ContentAnalysis;
use std::error::Error;

/// 标注数据（train_semantic_encoder 的输入）
#[derive(Debug, Clone)]
pub struct LabeledContent {
    pub raw: RawInput,
    pub true_weights: Vec<(u32, f32)>,    // 真实领域权重（监督信号）
}

pub trait ContentAnalyzer: Send + Sync {
    /// 主入口：输入原始文本 → 输出 ContentAnalysis
    fn analyze(&self, raw: &RawInput) -> Result<ContentAnalysis, Box<dyn Error>>;
    /// 返回所有注册领域（只读）
    fn domains(&self) -> &[Domain];
    /// 纯关键词匹配（不带 SyntaxPattern / MetaRouter，HybridAnalyzer 判断硬规则是否够用）
    fn rule_based_match(&self, _text: &str) -> Vec<(u32, f32)> { Vec::new() }
    /// 可选：训练内部 MiniMLP（架构 §5.3 接口定义 + §5.4 HybridAnalyzer 真实实现）
    fn train_semantic_encoder(&mut self, _training_data: &[LabeledContent]) {}
    /// FeatureEncoder 实际输出维度
    fn feature_dim(&self) -> usize;
}
```

---

# 七、utils 层（工具层）

## 7.1 src/utils/mini_mlp.rs（架构 §5.2 第 5 行：可选轻量语义编码）

```rust
//! MiniMLP：可训练轻量 MLP（Linear → GELU → Linear）
//!
//! 架构设计文档 §5.2 分析维度表第 5 行：
//!   "可选：轻量语义编码 —— char n-gram → Embedding → 轻量 MLP（2-3 层） —— 处理词汇变化"
//!
//! 完整实现：
//!   forward:  y = (x @ W1^T + b1).gelu() @ W2^T + b2
//!   参数初始化: Kaiming 正态（fan_in 归一化）
//!   train/eval 模式切换（Phase 1 无 Dropout，预留）
//!   parameters() 返回权重 Tensor（让 SparseParameterPool 注册）
//!
//! 为什么 MiniMLP 独立于 FeatureEncoder？
//!   FeatureEncoder 是静态 hash 编码器（不可训练），MiniMLP 是可训练投影层
//!   train_semantic_encoder() 时只训练 MiniMLP 的参数
//!   analyze 流程变成: hash_feats → (可选 MiniMLP 投影) → 输出 Tensor

use oyta_tensor::prelude::*;

pub struct MiniMLP {
    w1: Tensor<f32, Cpu>, b1: Tensor<f32, Cpu>,
    w2: Tensor<f32, Cpu>, b2: Tensor<f32, Cpu>,
    training: bool,
}

impl MiniMLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Kaiming 初始化：randn × sqrt(2/fan_in)
        let w1 = Self::scale_tensor(&Tensor::<f32, Cpu>::randn([hidden_dim, input_dim]), (2.0/input_dim as f32).sqrt());
        let w2 = Self::scale_tensor(&Tensor::<f32, Cpu>::randn([output_dim, hidden_dim]), (2.0/hidden_dim as f32).sqrt());
        Self { w1, b1: Tensor::<f32, Cpu>::zeros([hidden_dim]),
               w2, b2: Tensor::<f32, Cpu>::zeros([output_dim]), training: false }
    }

    pub fn forward(&self, input: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // Layer 1: h = gelu(input @ W1^T + b1)
        // Layer 2: out = h @ W2^T + b2
        let w1t = Self::transpose_tensor(&self.w1, 0, 1);
        let h = input.matmul(&w1t).unwrap();
        // bias 广播：[batch, hidden] += [hidden]
        let batch = input.shape().dims[0];
        let hidden = self.w1.shape().dims[0];
        let b1_b = Self::broadcast_vec(&self.b1, batch, hidden);
        let h = h.add(&b1_b).unwrap();
        let h = h.gelu().unwrap();
        // Layer 2
        let w2t = Self::transpose_tensor(&self.w2, 0, 1);
        let out = h.matmul(&w2t).unwrap();
        let out_dim = self.w2.shape().dims[0];
        let b2_b = Self::broadcast_vec(&self.b2, batch, out_dim);
        out.add(&b2_b).unwrap()
    }

    pub fn parameters(&self) -> Vec<&Tensor<f32, Cpu>> { vec![&self.w1, &self.b1, &self.w2, &self.b2] }
    pub fn param_count(&self) -> usize { /* 4 个 Tensor numel 求和 */ }
    pub fn train(&mut self) { self.training = true; }
    pub fn eval(&mut self)  { self.training = false; }
    pub fn w1_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.w1 }
    pub fn b1_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.b1 }
    pub fn w2_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.w2 }
    pub fn b2_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.b2 }

    // ===== 内部辅助：transpose + broadcast + scale =====
    fn transpose_tensor(t: &Tensor<f32, Cpu>, _d1: usize, _d2: usize) -> Tensor<f32, Cpu> { /* 手动 Vec transpose */ }
    fn broadcast_vec(b: &Tensor<f32, Cpu>, batch: usize, dim: usize) -> Tensor<f32, Cpu> { /* [dim] → [batch, dim] */ }
    fn scale_tensor(t: &Tensor<f32, Cpu>, scale: f32) -> Tensor<f32, Cpu> { /* data_slice read → scale → from_data 重建 */ }
}
```

## 7.2 src/utils/feature_encoder.rs

```rust
//! FeatureEncoder：文本 → 固定维度 Tensor
//!
//! 两条路径：
//!   无 MiniMLP: FNV-1a hash n-gram → 桶计数 → L2 归一化 → from_data 构造 Tensor
//!   有 MiniMLP: hash → [1, 256] → MiniMLP(256→128→num_domains) → L2 归一化 → 输出
//!
//! 训练 FeatureEncoder: train_semantic_encoder() 监督训练 MiniMLP 参数
//!   训练后 FeatureEncoder 的 MiniMLP 学到"各领域典型 n-gram 分布"

use oyta_tensor::prelude::*;
use crate::types::domain::Domain;
use crate::utils::mini_mlp::MiniMLP;

pub struct FeatureEncoder {
    pub output_dim: usize,                              // hash 输出维度（默认 256）
    mini_mlp: Option<MiniMLP>,                          // 可选可训练投影层
    mlp_output_dim: Option<usize>,                      // 投影后最终输出维度
}

impl FeatureEncoder {
    pub fn new(output_dim: usize) -> Self { /* 静态 hash 模式 */ }
    pub fn with_mlp(mut self, hidden_dim: Option<usize>, mlp_output_dim: Option<usize>) -> Self { /* 开启 MiniMLP 模式 */ }
    pub fn mini_mlp(&self) -> Option<&MiniMLP> { self.mini_mlp.as_ref() }
    pub fn mini_mlp_mut(&mut self) -> Option<&mut MiniMLP> { self.mini_mlp.as_mut() }
    pub fn effective_output_dim(&self) -> usize { self.mlp_output_dim.unwrap_or(self.output_dim) }
    pub fn train(&mut self) { if let Some(mlp) = &mut self.mini_mlp { mlp.train(); } }
    pub fn eval(&mut self)  { if let Some(mlp) = &mut self.mini_mlp { mlp.eval(); } }

    /// 编码：文本 → [1, effective_output_dim]（全调 oyta_tensor API）
    pub fn encode(&self, text: &str, _domains: &[Domain]) -> Tensor<f32, Cpu> {
        // Step 1: FNV-1a 64-bit hash char 2/3-gram → 桶计数
        let mut feats = vec![0.0f32; self.output_dim];
        let chars: Vec<char> = text.chars().collect();
        for n in [2usize, 3] {
            for window in chars.windows(n) {
                let ng: String = window.iter().collect();
                let mut hash: u64 = 0xcbf29ce484222325;
                for &b in ng.as_bytes() { hash ^= b as u64; hash = hash.wrapping_mul(0x100000001b3); }
                feats[(hash as usize) % self.output_dim] += 1.0;
            }
        }
        // Step 2: 可选 MiniMLP 投影
        let data: Vec<f32> = if let Some(mlp) = &self.mini_mlp {
            let hash_tensor = Tensor::<f32, Cpu>::from_data(feats.clone(), [1, self.output_dim]).unwrap();
            mlp.forward(&hash_tensor).data_slice().to_vec()
        } else { feats };
        // Step 3: L2 归一化 + Tensor 构造
        let norm_sq: f32 = data.iter().map(|v| v*v).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        let normalized: Vec<f32> = data.iter().map(|v| v / norm).collect();
        Tensor::<f32, Cpu>::from_data(normalized, [1, self.effective_output_dim()]).expect("encode Tensor 构造失败")
    }
}
```

## 7.3 src/utils/syntax_pattern.rs（架构 §5.2 第 2 行：句法/结构特征）

```rust
//! SyntaxPattern：结构化模式预编译（比扁平关键词更精准）
//!
//! 例子：
//!   "上证指数 3125.50 点 市盈率 8.2 成交量 123.45 亿"
//!     → 结构化元素: [数字+点] [数字+元/亿] [市盈率] [成交量] → finance 强信号
//!   "第 3 条 合同 条款 违反 民法典 第 5 条"
//!     → 结构化元素: [第+数字+条] [合同/条款/民法典] → legal 强信号
//!   "体温 37.8℃ 血压 120/80 mmHg 血糖 6.5 mmol/L"
//!     → 结构化元素: [数字+/+数字] [数字+mmol/L/℃/mmHg] → medical 强信号
//!
//! 预编译 HashMap<u32, Vec<Regex>>：构造时一次性编译，detect 时只跑 find_iter.count()

pub struct SyntaxPatternLibrary { /* 预编译的 15+ 条领域特定正则 */ }

impl SyntaxPatternLibrary {
    pub fn new() -> Self {
        // ===== 金融领域（domain_id=1）=====
        // "数字+亿元/元/万/亿" / "数字+点+数字"（股价） / "数字+%"（百分比）
        // "成交量/成交额/市值/市盈率/K线/上证/深证/纳斯达克"
        // "±数字+数字"（涨跌额）
        // ===== 医疗领域（domain_id=2）=====
        // "数字+/+数字"（血压/比值） / "数字+mmol/L/mg/dL/℃/mmHg"（医疗指标）
        // "体温/血压/血糖/心率/诊断/处方/病历"
        // ===== 法律领域（domain_id=3）=====
        // "第+数字+条/款/项/章"（法条引用） / "合同/条款/甲方/乙方/违约/民法典"
        // "YYYY年MM月DD日"（合同/判决日期）
        // ===== 教育领域（domain_id=4）=====
        // "第+数字+课/章/节"（课程章节） / "分数/成绩/学分/考试/论文/GPA"
    }

    /// 对任意文本跑所有正则 → 返回 [(domain_id, weighted_hit_score)]
    pub fn detect(&self, text: &str) -> Vec<(u32, f32)> { /* 每条正则 find_iter.count() × weight 累加 */ }
}
```

## 7.4 src/utils/meta_router.rs（架构 §5.2 第 3 行：元数据辅助路由）

```rust
//! MetaRouter：metadata + timestamp → 领域权重调整（增量加法，不覆盖原始权重）
//!
//! 四条规则全部可配置：
//!   1. 文件扩展名   —— ext=.pdf/.docx → legal, .xls/.xlsx → finance, .ppt → education
//!   2. 来源 URL     —— source=finance.sina.com.cn → finance, medical.net → medical
//!   3. 财报季时间戳 —— Q1(1-3)/Q2(4-6)/Q3(7-9)/Q4(10-12) 月 → finance +0.15
//!   4. 作者/机构    —— author=银行/证券 → finance, 医院 → medical, 法院 → legal

pub struct MetaRouter {
    ext_rules: HashMap<String, (u32, f32)>,           // "pdf" → (3, 0.2)
    source_rules: Vec<(String, u32, f32)>,            // ("finance", 1, 0.4)
}

impl MetaRouter {
    pub fn new() -> Self { /* 注册所有扩展名 + 来源域名 + 财报季 + 作者机构规则 */ }

    /// 辅助路由：原始权重 + metadata + timestamp → 调整后权重
    /// 权重增量加法（保持原始权重相对关系）→ 再次归一化
    pub fn adjust(&self, weights: &[(u32, f32)], metadata: &HashMap<String, String>, timestamp: u64) -> Vec<(u32, f32)> {
        // Step 1: 收集所有 bonus（扩展名 / 来源 URL / 财报季 / 作者机构）
        // Step 2: 原始权重 + bonus 增量
        // Step 3: 再次归一化（加分后权重和可能 ≠ 1）
        // Step 4: 过滤权重 ≤ 0 的领域
    }
}
```

## 7.5 src/utils/padding_mask.rs

```rust
//! PaddingMask：变长输入 mask 检测
//! 语义：0=有效位置, 1=padding 位置
//! shape: [num_samples, max_padded_length]

pub struct PaddingMask;

impl PaddingMask {
    pub fn detect(sample_lengths: &[usize], max_len: usize) -> Vec<Vec<f32>> {
        // 每个样本: t < len → 0.0 (有效), t >= len → 1.0 (padding)
        sample_lengths.iter().map(|&len| {
            (0..max_len).map(|t| if t < len { 0.0f32 } else { 1.0f32 }).collect()
        }).collect()
    }
    pub fn all_valid(num_samples: usize, max_len: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0f32; max_len]; num_samples]
    }
}
```

---

# 八、analyzers 层（实现层）

## 8.1 src/analyzers/rule_based.rs（三种分析维度完整集成）

```rust
//! RuleBasedAnalyzer：regex 关键词 + 句法结构化模式 + metadata 辅助路由
//!
//! 架构 §5.2 三种分析维度全部实现：
//!   ✅ 关键词/实体匹配 —— 预编译 regex HashMap（构造时一次性编译）
//!   ✅ 句法/结构特征   —— SyntaxPatternLibrary detect()
//!   ✅ 元数据辅助路由   —— MetaRouter adjust()
//!
//! 完整 analyze 流程：
//!   1. 关键词 hits/ln(len) → 静态得分
//!   2. SyntaxPatternLibrary.detect → 结构化加分
//!   3. 融合（关键词 × 0.7 + 结构化 × 0.3）
//!   4. MetaRouter.adjust(weights, metadata, timestamp) → metadata 辅助调整
//!   5. 全零 fallback（没有任何命中时激活权重最高的领域）
//!   6. 归一化 + Evidence 记录 + FeatureEncoder Tensor + PaddingMask

pub struct RuleBasedAnalyzer {
    domains: Vec<Domain>,
    keyword_regexes: HashMap<u32, Vec<Regex>>,          // 关键词词典预编译
    syntax_lib: SyntaxPatternLibrary,                    // 句法结构化正则预编译
    meta_router: MetaRouter,                             // metadata + timestamp 辅助
    feature_encoder: FeatureEncoder,                     // hash n-gram → Tensor
    config: RuleBasedConfig,                             // 融合权重配置
}

pub struct RuleBasedConfig {
    pub keyword_weight: f32,   // 默认 0.7
    pub syntax_weight: f32,    // 默认 0.3
}

impl RuleBasedAnalyzer {
    pub fn new(domains: Vec<Domain>) -> Self { /* 预编译 regex HashMap + SyntaxPatternLibrary + MetaRouter + FeatureEncoder(256) */ }
    pub fn with_config(domains, config, feature_dim) -> Self { /* 链式配置 */ }
}

impl ContentAnalyzer for RuleBasedAnalyzer {
    fn analyze(&self, raw_input: &RawInput) -> Result<ContentAnalysis, Box<dyn Error>> {
        // Step 1: 关键词 hits/ln(len) → 静态权重
        // Step 2: SyntaxPatternLibrary.detect(text) → 结构化加分
        // Step 3: 融合（keyword × 0.7 + syntax × 0.3）
        // Step 4: 全零 fallback → 激活权重最高领域
        // Step 5: MetaRouter.adjust(weights, metadata, timestamp)
        // Step 6: 归一化 + Evidence
        // Step 7: feature_encoder.encode(text, domains) → Tensor
        // Step 8: padding_mask（如果 sample_lengths 存在）
    }
    fn rule_based_match(&self, text: &str) -> Vec<(u32, f32)> { /* 只返回关键词匹配（不带 syntax/meta） */ }
    fn train_semantic_encoder(&mut self, _data: &[LabeledContent]) { /* RuleBasedAnalyzer 不训练（纯规则） */ }
    fn feature_dim(&self) -> usize { self.feature_encoder.effective_output_dim() }
}
```

## 8.2 src/analyzers/tfidf.rs

```rust
//! TFIDFAnalyzer：TF-IDF cosine 相似度 + SyntaxPattern + MetaRouter 集成
//!
//! 架构 §5.4 列的三种内置实现之一。HybridAnalyzer 硬规则不足时降级用它。
//! IDF 预构建：每个 n-gram 出现的领域数 → log(num_domains / df + 1)
//! 主题向量：每个领域的关键词 TF-IDF → hash 桶 → L2 归一化

pub struct TFIDFAnalyzer {
    domains: Vec<Domain>,
    domain_vectors: HashMap<u32, Vec<f32>>,    // 预构建主题向量
    idf: HashMap<String, f32>,                  // n-gram → IDF
    feature_encoder: FeatureEncoder,
    syntax_lib: SyntaxPatternLibrary,
    meta_router: MetaRouter,
    vector_dim: usize,
}

impl ContentAnalyzer for TFIDFAnalyzer {
    fn analyze(&self, raw: &RawInput) -> Result<ContentAnalysis, Box<dyn Error>> {
        // Step 1: 输入文本 → TF-IDF 向量 → 与每个领域主题向量 cosine → softmax
        // Step 2: SyntaxPatternLibrary.detect → 结构化加分（cosine × 0.7 + syntax × 0.3）
        // Step 3: MetaRouter.adjust(weights, metadata, timestamp)
        // Step 4: Evidence + FeatureEncoder + PaddingMask
    }
    fn rule_based_match(&self, _text: &str) -> Vec<(u32, f32)> { Vec::new() }  // TFIDF 没有硬规则
    fn train_semantic_encoder(&mut self, _data: &[LabeledContent]) {}
    fn feature_dim(&self) -> usize { self.feature_encoder.effective_output_dim() }
}
```

## 8.3 src/analyzers/hybrid.rs（三种内置实现的"生产首选" + MiniMLP 完整监督训练）

```rust
//! HybridAnalyzer：规则优先 + TFIDF 融合 + MiniMLP 监督训练
//!
//! 架构 §5.4 三种内置实现的"生产首选"：
//!   ✅ RuleBasedAnalyzer（已集成 SyntaxPattern + MetaRouter）
//!   ✅ TFIDFAnalyzer（已集成 SyntaxPattern + MetaRouter）
//!   ✅ train_semantic_encoder 完整实现（手动反向传播 CE→softmax→MLP 内部 + SGD 更新）
//!
//! 三种判断路径：
//!   1. 硬规则命中 ≥ 2 领域 → 直接用硬规则完整结果（跳过 TFIDF，最快）
//!   2. 硬规则命中 1 领域   → 硬规则 × 0.7 + TFIDF × 0.3 融合
//!   3. 硬规则零命中       → 纯 TFIDF

use anyhow::Result;

pub struct HybridAnalyzer {
    rule_based: RuleBasedAnalyzer,
    tfidf: TFIDFAnalyzer,
    config: HybridConfig,
    semantic_mlp: Option<MiniMLP>,       // 训练完保存的 MiniMLP
    semantic_lr: f32,
    num_domains: usize,
}

pub struct HybridConfig {
    pub hard_rule_domain_threshold: usize,   // 默认 2
    pub hard_weight_ratio: f32,              // 默认 0.7
    pub tfidf_weight_ratio: f32,             // 默认 0.3
}

impl HybridAnalyzer {
    pub fn new(domains: Vec<Domain>) -> Self { /* RuleBasedAnalyzer + TFIDFAnalyzer 一起初始化 */ }

    /// train_semantic_encoder_full：MiniMLP 监督训练（架构 §5.3 + §5.4 真实实现）
    ///
    /// 完整流程（手动反向传播 + SGD，不依赖 oyta-autograd）：
    ///   1. 初始化 MiniMLP(input=256, hidden=128, output=num_domains)
    ///   2. 训练 N 个 epoch：
    ///      a. hash_feats = 纯 FNV-1a hash（和 FeatureEncoder.encode 前半部分一致）
    ///      b. logits = MiniMLP.forward(hash_feats)  → [1, num_domains]
    ///      c. pred = softmax(logits)                 → [1, num_domains]
    ///      d. loss = -Σ true_weights[i] * log(pred[i] + eps)   （CrossEntropy）
    ///      e. dlogits = pred - true_weights          （CE + softmax 解析梯度）
    ///      f. mlp_backward(mini_mlp, hash_feats, dlogits, lr)  → 手动链式法则
    ///         Layer 2 反向: dW2, db2, dh_pre（GELU 梯度）
    ///         Layer 1 反向: dW1, db1
    ///      g. SGD: w -= lr * grad
    ///   3. 保存训练好的 MiniMLP 到 self.semantic_mlp
    pub fn train_semantic_encoder_full(&mut self, training_data: &[LabeledContent],
                                       epochs: Option<usize>, lr: Option<f32>) -> Result<()> {
        // 详见完整实现（手动反向传播公式 + SGD 更新）
    }

    /// MiniMLP 手动反向传播（链式法则，全 Vec 数学，不依赖 oyta_tensor autograd）
    fn mlp_backward(&mut self, mlp: &mut MiniMLP, hash_feats: &[f32],
                    _logits: &[f32], dlogits: &[f32], lr: f32) {
        // MiniMLP 结构: Linear(256→128) → GELU → Linear(128→num_domains)
        // Layer 2: dW2 = dh_act^T · dlogits, db2 = dlogits, dh_act = dlogits · W2^T
        // GELU: dh_pre = dh_act ⊙ gelu_grad(h_pre)
        // Layer 1: dW1 = hash_feats^T · dh_pre, db1 = dh_pre
        // GELU 用 tanh 近似（Rust 2024 稳定版兼容，业界标准）
        // gelu(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // gelu_grad: 解析链式法则

        // 读 W1, b1, W2, b2 数据 → 手动链式法则算梯度 → SGD 更新 → from_data 重建 Tensor
        // oyta_tensor 无 data_slice_mut，所以全走 data_slice read → 手动算 new data → from_data 写回
    }

    // ===== 纯函数辅助 =====
    fn pure_hash_feats(text: &str, dim: usize) -> Vec<f32> { /* 和 FeatureEncoder.encode 前半部分完全一致 */ }
    fn softmax_vec(logits: &[f32]) -> Vec<f32> { /* 数值稳定版（减 max） */ }
    fn gelu(x: f32) -> f32 { /* tanh 近似版 */ }
    fn gelu_grad(x: f32) -> f32 { /* tanh 近似的解析梯度 */ }
}

impl ContentAnalyzer for HybridAnalyzer {
    fn analyze(&self, raw: &RawInput) -> Result<ContentAnalysis, Box<dyn Error>> {
        // Step 1: 先跑硬规则（最快，确定命中多少领域）
        let hard_matches = self.rule_based.rule_based_match(&raw.text);
        let num_hit = hard_matches.iter().filter(|(_, w)| *w > 0.0).count();

        // 情况 1：硬规则命中 ≥ 2 领域 → 直接用硬规则完整结果（跳过 TFIDF）
        if num_hit >= self.config.hard_rule_domain_threshold {
            return self.rule_based.analyze(raw);
        }

        // 情况 2 & 3：需要 TFIDF 融合
        let hard_result = self.rule_based.analyze(raw)?;
        let tfidf_result = self.tfidf.analyze(raw)?;

        // Step 2: 加权融合（hard × 0.7 + tfidf × 0.3）
        let hard_map: HashMap<u32, f32> = hard_result.domain_weights.iter().copied().collect();
        let tfidf_map: HashMap<u32, f32> = tfidf_result.domain_weights.iter().copied().collect();
        let all_ids: HashSet<u32> = hard_map.keys().chain(tfidf_map.keys()).copied().collect();
        let mut fused: Vec<(u32, f32)> = Vec::new();
        for did in all_ids {
            let hw = hard_map.get(&did).copied().unwrap_or(0.0);
            let tw = tfidf_map.get(&did).copied().unwrap_or(0.0);
            let fw = hw * self.config.hard_weight_ratio + tw * self.config.tfidf_weight_ratio;
            if fw > 0.0 { fused.push((did, fw)); }
        }

        // Step 3: 归一化 + 全零 fallback
        let total: f32 = fused.iter().map(|(_, w)| *w).sum();
        let normalized: Vec<(u32, f32)> = if total > 0.0 {
            fused.iter().map(|(id, w)| (*id, w/total)).collect()
        } else {
            // fallback：权重最高的那个
            let best = fused.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            if let Some((id, _)) = best { vec![(*id, 1.0)] } else { fused }
        };

        let mut result = ContentAnalysis::new(normalized, hard_result.feature_vector.clone());
        result.evidence = hard_result.evidence.clone();
        result.padding_mask = hard_result.padding_mask.clone();
        Ok(result)
    }

    fn train_semantic_encoder(&mut self, data: &[LabeledContent]) {
        // 委托给 train_semantic_encoder_full（任何错误 eprintln 但不 panic）
        if let Err(e) = self.train_semantic_encoder_full(data, None, None) {
            eprintln!("[HybridAnalyzer.train_semantic_encoder] 训练失败: {}", e);
        }
    }

    fn domains(&self) -> &[Domain] { self.rule_based.domains() }
    fn rule_based_match(&self, text: &str) -> Vec<(u32, f32)> { self.rule_based.rule_based_match(text) }
    fn feature_dim(&self) -> usize { self.rule_based.feature_dim() }
}
```

---

# 九、验收标准（全部通过 22 个测试）

## 9.1 编译验收

```
✅ cargo check --package oyta-content 零 error 零 warning
✅ Cargo.toml 只写 oyta-tensor = "0.1.1"（无 path 依赖、无 rust-version）
✅ mod.rs 只做 pub mod 声明（types/mod.rs, traits/mod.rs, utils/mod.rs, analyzers/mod.rs）
✅ FeatureEncoder.encode 调 Tensor::<f32, Cpu>::from_data（正确调用 oyta-tensor）
✅ MiniMLP.forward 调 Tensor::matmul / .add / .gelu / .from_data
✅ 每行代码有中文注释
```

## 9.2 行为验收（9 条集成测试全部通过）

| 测试 | 覆盖 | 结果 |
|------|------|------|
| test_rule_based_finance_detection | "净利润同比增长" → finance > generic | ✅ |
| test_hybrid_medical_detection | 医疗文本 → medical > 0.3 | ✅ |
| test_tfidf_legal_detection | 法律文本 → legal > 0.15 | ✅ |
| test_empty_text_fallback | 空文本不 panic → fallback best domain = 1.0 | ✅ |
| test_metadata_router_pdf_legal | PDF 扩展名提升 legal 权重 | ✅ |
| test_feature_encoder_tensor_shape | Tensor shape = [1, feature_dim] | ✅ |
| test_train_semantic_encoder_completes | MiniMLP 训练完成 → 保存到 semantic_mlp | ✅ |
| test_evidence_present | Evidence 不为空 + contribution ≥ 0 | ✅ |
| test_variable_length_padding_mask | 变长 mask 语义正确（0=有效, 1=padding） | ✅ |

## 9.3 模块验收（13 条单元测试全部通过）

| 模块 | 测试 | 结果 |
|------|------|------|
| MiniMLP | forward_shape ✅ / param_count ✅ / basic_values ✅ | 3/3 |
| SyntaxPattern | finance ✅ / medical ✅ / legal ✅ / empty ✅ | 4/4 |
| MetaRouter | pdf_legal ✅ / source_finance ✅ / timestamp_april ✅ / empty_no_change ✅ | 4/4 |
| PaddingMask | detect ✅ / all_valid ✅ | 2/2 |

---

# 十、关键 API 速查

```rust
use oyta_content::prelude::*;

// ===== 构造 =====
let domains = default_domains();
let rule = RuleBasedAnalyzer::new(domains.clone());
let tfidf = TFIDFAnalyzer::new(domains.clone());
let hybrid = HybridAnalyzer::new(domains.clone());

// ===== 分析 =====
let raw = RawInput::from_text("净利润同比增长 15%").with_timestamp(1700000000);
let result: ContentAnalysis = hybrid.analyze(&raw).unwrap();
for (did, w) in &result.domain_weights {
    println!("domain {} weight {}", did, w);
}
println!("feature_vector shape: {:?}", result.feature_vector.shape().dims);

// ===== metadata 辅助 =====
let raw2 = RawInput::from_text("合同 条款 签订").with_metadata("ext", "pdf");
let r2 = hybrid.analyze(&raw2).unwrap();   // PDF 扩展名 → legal 权重更高

// ===== 变长输入 =====
let raw3 = RawInput::from_text("短").with_variable_lengths(vec![1, 10], 10);
let r3 = hybrid.analyze(&raw3).unwrap();
if let Some(mask) = &r3.padding_mask {
    // mask[0][0]=0.0（有效）, mask[0][1..]=1.0（padding）
}

// ===== MiniMLP 监督训练（可选）=====
let mut labeled: Vec<LabeledContent> = vec![
    LabeledContent { raw: RawInput::from_text("利润 营收 股价"), true_weights: vec![(1, 0.9), (0, 0.1)] },
];
hybrid.train_semantic_encoder(&labeled);

// ===== FeatureEncoder MiniMLP 模式 =====
let mut fe = FeatureEncoder::new(256).with_mlp(Some(128), Some(5));  // 投影到 5 维（= num_domains）
fe.train();
let tensor = fe.encode("净利润增长", &domains);
```
