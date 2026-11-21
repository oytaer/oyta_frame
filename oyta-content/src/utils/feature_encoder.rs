//! FeatureEncoder：把任意文本编码成固定维度的特征 Tensor
//!
//! 架构文档 §5.2 分析维度表第 5 行：
//!   "可选：轻量语义编码 —— char n-gram → Embedding → 轻量 MLP（2-3 层） —— 处理词汇变化"
//!
//! 完整实现：
//!   默认模式（无 MiniMLP）：
//!     hash_feats = FNV-1a hash n-gram → 桶计数 → L2 归一化
//!     输出 Tensor shape = [1, output_dim]（静态，不可训练）
//!
//!   训练模式（带 MiniMLP）：
//!     hash_feats → MiniMLP(hidden → mlp_output_dim) → L2 归一化 → 输出
//!     输出 Tensor shape = [1, mlp_output_dim]（MiniMLP 可训练）
//!
//!   构造 MiniMLP 的意义：
//!     静态 hash 编码器是"通用"的，对所有领域输出同一种特征分布
//!     加上 MiniMLP 后，可以让 train_semantic_encoder() 学到
//!     "金融领域的典型 n-gram 分布"、"医疗领域的典型 n-gram 分布"
//!     让不同领域的输入特征在投影空间里自然聚类 → 路由器更准

use oyta_tensor::prelude::*;
use crate::types::domain::Domain;
use crate::utils::mini_mlp::MiniMLP;

pub struct FeatureEncoder {
    /// hash 输出维度（静态桶数量）
    pub output_dim: usize,
    /// 可选：可训练 MiniMLP 投影层（None = 静态 hash 模式）
    /// Some(mini_mlp) 时 encode 变成 hash → mlp → 归一化
    mini_mlp: Option<MiniMLP>,
    /// MiniMLP 投影后输出维度（None 时 = output_dim）
    mlp_output_dim: Option<usize>,
}

impl FeatureEncoder {
    /// 构造：默认静态 hash 模式（无 MiniMLP，不可训练）
    pub fn new(output_dim: usize) -> Self {
        Self { output_dim, mini_mlp: None, mlp_output_dim: None }
    }

    /// 开启 MiniMLP 投影模式（可训练）
    ///
    /// # 参数
    /// - hidden_dim: MiniMLP 隐藏层维度（默认 output_dim / 2）
    /// - mlp_output_dim: 投影后最终输出维度（默认 output_dim，保持维度不变）
    pub fn with_mlp(mut self, hidden_dim: Option<usize>, mlp_output_dim: Option<usize>) -> Self {
        let hidden = hidden_dim.unwrap_or(self.output_dim / 2).max(16);
        let out = mlp_output_dim.unwrap_or(self.output_dim);
        self.mini_mlp = Some(MiniMLP::new(self.output_dim, hidden, out));
        self.mlp_output_dim = Some(out);
        self
    }

    /// MiniMLP 参数访问（让 train_semantic_encoder / SparseParameterPool 注册）
    pub fn mini_mlp(&self) -> Option<&MiniMLP> { self.mini_mlp.as_ref() }
    pub fn mini_mlp_mut(&mut self) -> Option<&mut MiniMLP> { self.mini_mlp.as_mut() }

    /// 当前 encode 的实际输出维度
    pub fn effective_output_dim(&self) -> usize { self.mlp_output_dim.unwrap_or(self.output_dim) }

    /// MiniMLP train/eval 模式切换
    pub fn train(&mut self) { if let Some(mlp) = &mut self.mini_mlp { mlp.train(); } }
    pub fn eval(&mut self) { if let Some(mlp) = &mut self.mini_mlp { mlp.eval(); } }

    /// 编码：文本 → [1, effective_output_dim] 固定维度 Tensor
    ///
    /// 两条路径：
    ///   无 MiniMLP: hash → L2 归一化 → from_data
    ///   有 MiniMLP: hash → Tensor → MiniMLP → L2 归一化 → 输出
    pub fn encode(&self, text: &str, _domains: &[Domain]) -> Tensor<f32, Cpu> {
        // ===== Step 1: FNV-1a hash n-gram → 桶计数 =====
        let mut feats = vec![0.0f32; self.output_dim];
        let chars: Vec<char> = text.chars().collect();
        for n in [2usize, 3] {
            for window in chars.windows(n) {
                let ng: String = window.iter().collect();
                let hash = Self::fnv1a_64(ng.as_bytes());
                let idx = (hash as usize) % self.output_dim;
                feats[idx] += 1.0;
            }
        }

        // ===== Step 2: 如果有 MiniMLP → 先投影 =====
        let data: Vec<f32> = if let Some(mlp) = &self.mini_mlp {
            // hash feats → [1, output_dim] Tensor → MiniMLP → [1, mlp_output_dim] Tensor
            let hash_tensor = Tensor::<f32, Cpu>::from_data(feats, [1, self.output_dim])
                .expect("hash Tensor 构造失败");
            let projected = mlp.forward(&hash_tensor);
            projected.data_slice().to_vec()
        } else {
            feats
        };

        // ===== Step 3: L2 归一化（消除文本长度影响）=====
        let norm_sq: f32 = data.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-8);  // 防除零
        let normalized: Vec<f32> = data.iter().map(|v| v / norm).collect();

        // ===== Step 4: 构造最终 Tensor（调 oyta_tensor）=====
        let dim = self.effective_output_dim();
        Tensor::<f32, Cpu>::from_data(normalized, [1, dim])
            .expect("FeatureEncoder::encode Tensor 构造失败")
    }

    /// FNV-1a 64-bit hash（标准实现，纯 Rust）
    /// offset_basis = 0xcbf29ce484222325, prime = 0x100000001b3
    fn fnv1a_64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}
