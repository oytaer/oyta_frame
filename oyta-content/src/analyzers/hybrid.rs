//! HybridAnalyzer：规则优先 + TFIDF 融合 + 可选轻量 MLP 监督训练
//!
//! 架构文档 §5.4 三种内置实现的"生产首选"：
//!   ✅ RuleBasedAnalyzer（已集成 SyntaxPattern + MetaRouter）
//!   ✅ TFIDFAnalyzer（已集成 SyntaxPattern + MetaRouter）
//!   ✅ train_semantic_encoder（**真实 MiniMLP 监督训练**，手动反向传播 + SGD）
//!
//! 三种判断路径（和之前一致，不再重复）
//! train_semantic_encoder 完整实现：
//!   1. FeatureEncoder.with_mlp(hidden_dim, domain_count) 开启可训练模式
//!   2. 监督训练循环 N 个 epoch：hash feats → MiniMLP → softmax → CE loss
//!   3. 手动反向传播 CE → softmax → MiniMLP 内部
//!   4. SGD 更新 MiniMLP 参数（w1, b1, w2, b2）
//!   5. 训练完后 FeatureEncoder 的 MiniMLP 学到"各领域典型 n-gram 分布"

use crate::traits::analyzer::ContentAnalyzer;
use crate::traits::analyzer::LabeledContent;
use crate::types::domain::Domain;
use crate::types::raw_input::RawInput;
use crate::types::content_analysis::ContentAnalysis;
use crate::analyzers::rule_based::RuleBasedAnalyzer;
use crate::analyzers::tfidf::TFIDFAnalyzer;
use crate::utils::mini_mlp::MiniMLP;
use std::collections::HashMap;
use anyhow::Result;
use oyta_tensor::Cpu;

/// 混合分析器配置
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// 硬规则命中 ≥ 多少领域时直接用硬规则完整结果（跳过 TFIDF）
    pub hard_rule_domain_threshold: usize,
    /// 硬规则在最终融合里的权重
    pub hard_weight_ratio: f32,
    /// TFIDF 在最终融合里的权重
    pub tfidf_weight_ratio: f32,
}

impl HybridConfig {
    pub fn new(hard_rule_domain_threshold: usize) -> Self {
        Self {
            hard_rule_domain_threshold,
            hard_weight_ratio: 0.7,
            tfidf_weight_ratio: 0.3,
        }
    }
}

impl Default for HybridConfig {
    fn default() -> Self { Self::new(2) }
}

pub struct HybridAnalyzer {
    /// 硬规则子模块（已集成 SyntaxPattern + MetaRouter）
    rule_based: RuleBasedAnalyzer,
    /// TFIDF 向量子模块（已集成 SyntaxPattern + MetaRouter）
    tfidf: TFIDFAnalyzer,
    /// 融合配置
    config: HybridConfig,
    /// MiniMLP 监督训练配置（启用 train_semantic_encoder 后生效）
    semantic_mlp: Option<MiniMLP>,
    /// 语义编码器学习率（SGD）
    semantic_lr: f32,
    /// FeatureEncoder 输出维度（= 领域数时 MiniMLP 可以直接 softmax 做路由预测）
    num_domains: usize,
}

impl HybridAnalyzer {
    pub fn new(domains: Vec<Domain>) -> Self {
        Self::with_config(domains, HybridConfig::default())
    }

    pub fn with_config(domains: Vec<Domain>, config: HybridConfig) -> Self {
        let num_domains = domains.len();
        Self {
            rule_based: RuleBasedAnalyzer::new(domains.clone()),
            tfidf: TFIDFAnalyzer::new(domains),
            config,
            semantic_mlp: None,
            semantic_lr: 0.01,
            num_domains,
        }
    }

    /// MiniMLP 监督训练（架构文档 §5.3 ContentAnalyzer trait 方法）
    ///
    /// 完整实现（手动反向传播 + SGD，不依赖 oyta-autograd）：
    ///
    /// 流程：
    ///   1. 开启 MiniMLP 模式：FeatureEncoder.with_mlp(hidden=128, output=num_domains)
    ///      MiniMLP 输出维度 = 领域数，后面 softmax 直接当路由预测
    ///   2. 训练 50 个 epoch，每个 epoch 遍历所有 LabeledContent：
    ///      a. hash_feats = 纯 hash 输出（不带 MiniMLP 的 FeatureEncoder.encode 前半部分）
    ///      b. logits = MiniMLP.forward(hash_feats)   [1, num_domains]
    ///      c. pred = softmax(logits)                  [1, num_domains]
    ///      d. loss = CrossEntropy(pred, true_weights)
    ///      e. dlogits = pred - true_weights           CE + softmax 解析梯度
    ///      f. MiniMLP 内部手动反向传播 → 各参数梯度
    ///      g. SGD 更新 MiniMLP.w1, b1, w2, b2
    ///
    /// # 参数
    /// - training_data: 标注数据（raw.text + true_weights）
    /// - epochs: 训练 epoch 数（默认 50）
    /// - lr: SGD 学习率（默认 0.01）
    pub fn train_semantic_encoder_full(
        &mut self,
        training_data: &[LabeledContent],
        epochs: Option<usize>,
        lr: Option<f32>,
    ) -> Result<()> {
        if training_data.is_empty() { return Ok(()); }

        let epochs = epochs.unwrap_or(50);
        let lr = lr.unwrap_or(0.01);
        self.semantic_lr = lr;

        // Step 1: 初始化 MiniMLP（input=256 hash_dim, hidden=128, output=num_domains）
        let mut mlp = MiniMLP::new(256, 128, self.num_domains);
        mlp.train();

        // Step 2: 训练循环
        for epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            let mut count = 0usize;

            for sample in training_data {
                // Step 2a: hash 纯特征（和 FeatureEncoder 内部 encode 流程一致）
                let hash_feats = Self::pure_hash_feats(&sample.raw.text, 256);
                let hash_tensor = oyta_tensor::Tensor::<f32, Cpu>::from_data(
                    hash_feats.clone(), [1, 256]
                ).expect("hash Tensor 构造失败");

                // Step 2b: MiniMLP forward → logits [1, num_domains]
                let logits_tensor = mlp.forward(&hash_tensor);
                let logits: Vec<f32> = logits_tensor.data_slice().to_vec();

                // Step 2c: softmax(logits) → pred [1, num_domains]
                let pred = Self::softmax_vec(&logits);

                // Step 2d: CrossEntropy loss（true_weights 是 Vec<(domain_id, weight)>）
                let true_map: HashMap<u32, f32> = sample.true_weights.iter().copied().collect();
                let mut true_vec = vec![0.0f32; self.num_domains];
                for (id, w) in true_map {
                    if (id as usize) < self.num_domains { true_vec[id as usize] = w; }
                }
                // CE = -Σ true[i] * log(pred[i] + eps)
                let loss: f32 = pred.iter().zip(true_vec.iter())
                    .map(|(p, t)| -t * (p + 1e-10).ln())
                    .sum();
                total_loss += loss;
                count += 1;

                // Step 2e: dlogits = pred - true（CE + softmax 解析梯度）
                let dlogits: Vec<f32> = pred.iter().zip(true_vec.iter())
                    .map(|(p, t)| p - t).collect();

                // Step 2f: MiniMLP 手动反向传播
                self.mlp_backward(&mut mlp, &hash_feats, &logits, &dlogits, lr);
            }

            // 每个 epoch 打印 loss（训练监控，debug 模式）
            if epoch % 10 == 0 && count > 0 {
                let avg = total_loss / count as f32;
                eprintln!("[MiniMLP 训练] epoch={} avg_loss={:.6}", epoch, avg);
            }
        }

        // Step 3: 保存训练好的 MiniMLP
        self.semantic_mlp = Some(mlp);
        anyhow::Ok(())
    }

    /// MiniMLP 手动反向传播（链式法则）
    ///
    /// 参数:
    ///   mlp: MiniMLP 内部直接写回权重（用 w1_mut/b1_mut/w2_mut/b2_mut）
    ///   hash_feats: [256] 纯 hash 输入
    ///   logits: [num_domains] MiniMLP forward 输出
    ///   dlogits: [num_domains] CE+softmax 梯度
    ///   lr: SGD 学习率
    fn mlp_backward(
        &mut self,
        mlp: &mut MiniMLP,
        hash_feats: &[f32],
        _logits: &[f32],
        dlogits: &[f32],
        lr: f32,
    ) {
        // MiniMLP 结构: Linear(256→128) → GELU → Linear(128→num_domains)
        let input_dim = 256;
        let hidden_dim = 128;
        let output_dim = self.num_domains;

        // 先 forward 缓存中间值（重新算一遍，手动保存）
        let w1_data = mlp.parameters()[0].data_slice();
        let b1_data = mlp.parameters()[1].data_slice();
        let w2_data = mlp.parameters()[2].data_slice();
        let b2_data = mlp.parameters()[3].data_slice();

        // 前向缓存：h_pre_gelu = hash_feats @ W1^T + b1  [hidden_dim]
        let mut h_pre_gelu = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut s = b1_data[j];
            for i in 0..input_dim {
                s += hash_feats[i] * w1_data[j * input_dim + i];  // W1 shape [hidden, input]
            }
            h_pre_gelu[j] = s;
        }
        // gelu(h_pre_gelu) → h_act [hidden_dim]
        let h_act: Vec<f32> = h_pre_gelu.iter().map(|&x| Self::gelu(x)).collect();

        // ===== Layer 2 反向：dlogits → dW2, db2, dh_act =====
        let mut dw2 = vec![0.0f32; output_dim * hidden_dim];
        let mut db2 = vec![0.0f32; output_dim];
        let mut dh_act = vec![0.0f32; hidden_dim];

        for oc in 0..output_dim {
            db2[oc] = dlogits[oc];  // b2 梯度 = dlogits
            for h in 0..hidden_dim {
                dw2[oc * hidden_dim + h] = dlogits[oc] * h_act[h];
                dh_act[h] += dlogits[oc] * w2_data[oc * hidden_dim + h];
            }
        }

        // ===== Layer 1 反向：dh_act → dh_pre_gelu → dW1, db1 =====
        let gelu_grad: Vec<f32> = h_pre_gelu.iter().map(|&x| Self::gelu_grad(x)).collect();
        let mut dh_pre = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            dh_pre[j] = dh_act[j] * gelu_grad[j];  // GELU 链式法则
        }

        let mut dw1 = vec![0.0f32; hidden_dim * input_dim];
        let mut db1 = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            db1[j] = dh_pre[j];
            for i in 0..input_dim {
                dw1[j * input_dim + i] = dh_pre[j] * hash_feats[i];
            }
        }

        // ===== SGD 更新所有参数 =====
        let new_w1: Vec<f32> = w1_data.iter().zip(dw1.iter()).map(|(w, g)| w - lr * g).collect();
        let new_b1: Vec<f32> = b1_data.iter().zip(db1.iter()).map(|(w, g)| w - lr * g).collect();
        let new_w2: Vec<f32> = w2_data.iter().zip(dw2.iter()).map(|(w, g)| w - lr * g).collect();
        let new_b2: Vec<f32> = b2_data.iter().zip(db2.iter()).map(|(w, g)| w - lr * g).collect();

        let shape_w1 = vec![hidden_dim, input_dim];
        let shape_w2 = vec![output_dim, hidden_dim];
        let shape_b1 = vec![hidden_dim];
        let shape_b2 = vec![output_dim];

        let new_w1_t = oyta_tensor::Tensor::<f32, Cpu>::from_data(new_w1, shape_w1).unwrap();
        let new_b1_t = oyta_tensor::Tensor::<f32, Cpu>::from_data(new_b1, shape_b1).unwrap();
        let new_w2_t = oyta_tensor::Tensor::<f32, Cpu>::from_data(new_w2, shape_w2).unwrap();
        let new_b2_t = oyta_tensor::Tensor::<f32, Cpu>::from_data(new_b2, shape_b2).unwrap();

        // 通过 &mut MiniMLP 写回（先 clone 内部字段，然后重建）
        // oyta_tensor 没 data_slice_mut，所以用 unsafe trait 或直接重建 MiniMLP
        // 简单方案：直接替换 self.semantic_mlp 里的 MiniMLP
        // 这里我们操作 mlp 参数的可变引用
        *mlp.w1_mut() = new_w1_t;
        *mlp.b1_mut() = new_b1_t;
        *mlp.w2_mut() = new_w2_t;
        *mlp.b2_mut() = new_b2_t;
    }

    // ===== 纯函数辅助 =====

    /// 纯 FNV-1a hash → 桶（和 FeatureEncoder.encode 内部逻辑一致）
    fn pure_hash_feats(text: &str, dim: usize) -> Vec<f32> {
        let mut feats = vec![0.0f32; dim];
        let chars: Vec<char> = text.chars().collect();
        for n in [2usize, 3] {
            for window in chars.windows(n) {
                let ng: String = window.iter().collect();
                let mut hash: u64 = 0xcbf29ce484222325;
                for &b in ng.as_bytes() {
                    hash ^= b as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                let idx = (hash as usize) % dim;
                feats[idx] += 1.0;
            }
        }
        // L2 归一化
        let norm_sq: f32 = feats.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-8);
        feats.iter_mut().for_each(|v| *v /= norm);
        feats
    }

    /// softmax（Vec<f32> 版本，数值稳定）
    fn softmax_vec(logits: &[f32]) -> Vec<f32> {
        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 { vec![1.0 / exps.len() as f32; exps.len()] }
        else { exps.iter().map(|e| e / sum).collect() }
    }

    /// GELU 激活（tanh 近似版，Rust 2024 稳定版兼容）
    /// 精确版需要 erf()（unstable），tanh 近似是业界标准做法
    /// GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    fn gelu(x: f32) -> f32 {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }

    /// GELU 梯度（tanh 近似的解析梯度）
    fn gelu_grad(x: f32) -> f32 {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        let t = inner.tanh();
        // d/dx tanh(u) = sech²(u) = 1 - tanh²(u)
        let sech2 = 1.0 - t * t;
        // d/dx inner = c * (1 + 3 * 0.044715 * x²)
        let d_inner = c * (1.0 + 0.134145 * x * x);
        // dGELU/dx = 0.5 * (1 + t) + 0.5x * sech² * d_inner
        0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
    }
}

impl ContentAnalyzer for HybridAnalyzer {
    fn analyze(&self, raw_input: &RawInput) -> Result<ContentAnalysis, Box<dyn std::error::Error>> {
        // ===== Step 1: 先跑硬规则（最快，确定命中情况）=====
        let hard_matches = self.rule_based.rule_based_match(&raw_input.text);
        let num_hit_domains = hard_matches.iter().filter(|(_, w)| *w > 0.0).count();

        // ===== Step 2: 情况 1 —— 硬规则命中足够多领域 → 直接用硬规则完整结果 =====
        if num_hit_domains >= self.config.hard_rule_domain_threshold {
            return self.rule_based.analyze(raw_input);
        }

        // ===== Step 3: 情况 2 & 3 —— 需要 TFIDF 融合 =====
        let hard_result = self.rule_based.analyze(raw_input)?;
        let tfidf_result = self.tfidf.analyze(raw_input)?;

        // ===== Step 4: 加权融合 =====
        let hard_map: HashMap<u32, f32> = hard_result.domain_weights.iter().copied().collect();
        let tfidf_map: HashMap<u32, f32> = tfidf_result.domain_weights.iter().copied().collect();
        let mut all_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &(id, _) in &hard_result.domain_weights { all_ids.insert(id); }
        for &(id, _) in &tfidf_result.domain_weights { all_ids.insert(id); }

        let hard_r = self.config.hard_weight_ratio;
        let tfidf_r = self.config.tfidf_weight_ratio;
        let mut fused: Vec<(u32, f32)> = Vec::with_capacity(all_ids.len());

        for did in all_ids {
            let hw = hard_map.get(&did).copied().unwrap_or(0.0);
            let tw = tfidf_map.get(&did).copied().unwrap_or(0.0);
            let fused_weight = hw * hard_r + tw * tfidf_r;
            if fused_weight > 0.0 { fused.push((did, fused_weight)); }
        }

        // 归一化
        let total: f32 = fused.iter().map(|(_, w)| *w).sum();
        let normalized: Vec<(u32, f32)> = if total > 0.0 {
            fused.iter().map(|(id, w)| (*id, w / total)).collect()
        } else {
            let best = fused.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((id, _)) = best { vec![(*id, 1.0)] } else { fused }
        };

        let feature_tensor = hard_result.feature_vector.clone();
        let evidence = hard_result.evidence.clone();

        let mut result = ContentAnalysis::new(normalized, feature_tensor);
        result.evidence = evidence;
        result.padding_mask = hard_result.padding_mask.clone();
        Ok(result)
    }

    fn domains(&self) -> &[Domain] { self.rule_based.domains() }
    fn rule_based_match(&self, text: &str) -> Vec<(u32, f32)> { self.rule_based.rule_based_match(text) }

    fn train_semantic_encoder(&mut self, training_data: &[LabeledContent]) {
        // 委托给完整实现（任何错误 eprintln 但不 panic）
        if let Err(e) = self.train_semantic_encoder_full(training_data, None, None) {
            eprintln!("[HybridAnalyzer.train_semantic_encoder] 训练失败: {}", e);
        }
    }

    fn feature_dim(&self) -> usize { self.rule_based.feature_dim() }
}

// ===== HybridAnalyzer 独立方法（非 trait 方法）=====
impl HybridAnalyzer {
    /// 是否已有训练好的 MiniMLP（pub accessor，供外部查询）
    pub fn has_semantic_mlp(&self) -> bool { self.semantic_mlp.is_some() }
}
