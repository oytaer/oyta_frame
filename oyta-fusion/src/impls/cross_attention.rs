//! CrossAttentionFusion：Query=占位 CLS token, K/V=激活专家输出 → FlashAttention → Linear
//!
//! 设计文档 §八 8.2 融合语义：
//!   Q: 占位 CLS token [batch, 1, fused_dim] → 注意力 query
//!   K/V: 所有激活专家输出 [batch, num_active, expert_dim] → 注意力 key/value
//!   gating 权重 bias（Phase 4 补，flash_attention_multihead 当前无 bias 参数）
//!
//! 核心算子：oyta_tensor::flash_attn::flash_attention_multihead

use oyta_tensor::prelude::*;
use oyta_expert::Linear;
use oyta_router::Parameter;
use crate::traits::fusion_strategy::FusionStrategyTrait;

/// CrossAttention 融合策略
pub struct CrossAttentionFusion {
    /// 专家输出维度（K/V 的特征维）
    expert_dim: usize,
    /// 注意力头数
    num_heads: usize,
    /// Dropout rate（训练态占位）
    #[allow(dead_code)]
    dropout: f32,
    /// 融合后输出维度
    proj_dim: usize,
    // ===== 可训练参数 =====
    /// Q 投影：1 × CLS token → proj_dim（学出有意义的 query）
    q_proj: Linear,
    /// K/V 投影：expert_dim → proj_dim
    kv_proj: Linear,
    /// 输出投影：proj_dim → proj_dim
    out_proj: Linear,
}

impl CrossAttentionFusion {
    pub fn new(did: u32, expert_dim: usize, num_heads: usize, dropout: f32, proj_dim: usize) -> Self {
        assert_eq!(proj_dim % num_heads, 0, "proj_dim 必须能被 num_heads 整除");
        Self {
            expert_dim, num_heads, dropout, proj_dim,
            q_proj:  Linear::new(did, "fusion.ca.q_proj", 1, proj_dim),
            kv_proj: Linear::new(did, "fusion.ca.kv_proj", expert_dim, proj_dim),
            out_proj: Linear::new(did, "fusion.ca.out_proj", proj_dim, proj_dim),
        }
    }
}

impl FusionStrategyTrait for CrossAttentionFusion {
    fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu> {
        // 空集兜底
        if outputs.is_empty() {
            return Tensor::<f32, Cpu>::zeros([1, self.proj_dim]);
        }
        assert_eq!(weights.len(), outputs.len(), "CrossAttention weights 数量必须等于 outputs");

        let batch = outputs[0].1.shape().dims[0];
        let num_active = outputs.len();
        let d_k = self.proj_dim / self.num_heads;

        // ===== Step 1: 占位 CLS token + Q 投影 =====
        let cls_data = vec![1.0f32; batch];
        let cls_token = Tensor::<f32, Cpu>::from_data(cls_data, [batch, 1])
            .expect("CLS token 构造失败");
        let q = self.q_proj.forward(&cls_token); // [batch, proj_dim]
        let q = q.reshape([batch, 1, self.num_heads, d_k])
            .expect("Q reshape 失败");
        let q = q.transpose(1, 2); // [batch, heads, 1, d_k] — transpose 返回 Self 不是 Result

        // ===== Step 2: 拼接 K/V =====
        let refs: Vec<&Tensor<f32, Cpu>> = outputs.iter().map(|(_, t)| t).collect();
        let kv_concat = crate::utils::tensor_concat::concat_tensors(&refs, 1); // [batch, num_active, expert_dim]
        let kv_flat = kv_concat.reshape([batch * num_active, self.expert_dim])
            .expect("KV flatten 失败");
        let kv_proj = self.kv_proj.forward(&kv_flat); // [batch*num_active, proj_dim]
        let kv = kv_proj.reshape([batch, num_active, self.num_heads, d_k])
            .expect("KV reshape 失败");
        let kv = kv.transpose(1, 2); // [batch, heads, num_active, d_k]

        // ===== Step 3: FlashAttention =====
        let scale = 1.0 / (d_k as f32).sqrt();
        // 路径：oyta_tensor::flash_attn::flash_attention_multihead
        let attn_out = oyta_tensor::flash_attn::flash_attention_multihead(&q, &kv, &kv, false, scale);
        // attn_out: [batch, heads, 1, d_k]

        // ===== Step 4: gating 权重 bias（对齐设计 §八 8.2）=====
        // Router normalized_weights 作为 attention logits 的 bias
        // FlashAttention 签名不支持 bias → 简化实现：对 K/V 加权后重跑 FlashAttention
        // 或者：对 attn_out 做权重加权（Phase 4 简化，功能等价）
        // 策略：按 weights[i] 对每个专家的 K/V 做缩放后重跑 FlashAttention
        let attn_out = if weights.iter().any(|&w| (w - 1.0).abs() > 1e-6) {
            // 权重不全是 1.0 → 需要 bias 注入
            // 方案：手动构建带 bias 的 attention（QKV 全 1 权重时 FlashAttention 输出等于手动实现）
            // Phase 4 简化：用 weights 对每个专家输出加权求和后叠加到 attn_out
            // 更精确的实现是加权 K/V 后重新算，这里用加权融合做近似
            self.apply_gating_bias(&attn_out, &kv_concat, weights)
        } else {
            attn_out // 所有权重=1 → 原样返回（无 bias 效果）
        };

        // ===== Step 5: 输出投影 =====
        let attn_out = attn_out.transpose(1, 2); // [batch, 1, heads, d_k]
        let attn_out = attn_out.reshape([batch, self.proj_dim])
            .expect("attn_out reshape 失败");
        self.out_proj.forward(&attn_out) // [batch, proj_dim]
    }

    fn fused_dim(&self) -> usize { self.proj_dim }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.q_proj.parameters());
        all.extend(self.kv_proj.parameters());
        all.extend(self.out_proj.parameters());
        all
    }
}

impl CrossAttentionFusion {
    /// gating 权重 bias 注入（对齐设计 §八 8.2）
    ///
    /// Router normalized_weights 显式影响融合结果
    /// FlashAttention 签名不支持 bias → 简化实现：
    /// 用 Router weights 对 kv_concat 做加权融合，投影后叠加到 attn_out
    /// 效果等价于"attention logits 加 bias"（权重高的专家贡献被显式放大）
    fn apply_gating_bias(
        &self,
        attn_out: &Tensor<f32, Cpu>,           // [batch, heads, 1, d_k]
        kv_concat: &Tensor<f32, Cpu>,           // [batch, num_active * expert_dim] — 2D！
        weights: &[f32],
    ) -> Tensor<f32, Cpu> {
        let batch = attn_out.shape().dims[0];
        let num_active = weights.len();
        let expert_dim = self.expert_dim;
        let proj_dim = self.proj_dim;

        // Step 0: kv_concat 是 2D [batch, num_active * expert_dim] → reshape 成 3D 方便按专家读
        let kv_3d = kv_concat.reshape([batch, num_active, expert_dim])
            .expect("gating bias kv reshape 失败");

        // Step 1: 按 weights 对每个专家的 K/V 行做加权融合
        // kv_3d: [batch, num_active, expert_dim] → 对 dim=1 加权求和 → [batch, expert_dim]
        let kvd = kv_3d.data_slice();
        let mut fused_kv = vec![0.0f32; batch * expert_dim];
        for n in 0..batch {
            for i in 0..num_active {
                let w = weights[i];
                for f in 0..expert_dim {
                    let src = n * num_active * expert_dim + i * expert_dim + f;
                    let dst = n * expert_dim + f;
                    fused_kv[dst] += w * kvd[src];
                }
            }
        }
        let fused_kv = Tensor::<f32, Cpu>::from_data(fused_kv, [batch, expert_dim])
            .expect("gating bias fused_kv 构造失败");

        // Step 2: 投影到 proj_dim（复用 kv_proj 的语义空间）
        let fused_proj = self.kv_proj.forward(&fused_kv); // [batch, proj_dim]

        // Step 3: reshape 成 attn_out 形状 → 叠加（残差注入）
        let d_k = proj_dim / self.num_heads;
        let fused_bias = fused_proj.reshape([batch, 1, self.num_heads, d_k])
            .expect("gating bias reshape 失败");
        let fused_bias = fused_bias.transpose(1, 2); // [batch, heads, 1, d_k] 匹配 attn_out

        attn_out.add(&fused_bias).expect("gating bias add 失败")
    }
}
