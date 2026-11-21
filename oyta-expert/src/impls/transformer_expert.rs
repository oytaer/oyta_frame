//! TransformerExpert：Pre-LN Transformer（完整 7 组参数）
//!
//! 设计文档 §九 骨架结构：
//!   LN1 → QKV → FlashAttention → o_proj → 残差 → LN2 → FFN1 → GELU → FFN2 → 残差 → proj
//!
//! 核心算子：
//!   - FlashAttention：oyta_ops::attention::flash::flash_attention_multihead
//!   - LayerNorm：Tensor::layernorm
//!   - Linear：Linear::forward（内部 matmul + add）

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::traits::domain_expert::DomainExpert;
use crate::utils::linear::Linear;
use crate::utils::norm::LayerNorm;

/// Pre-LN Transformer 简化骨架（完整参数结构，FlashAttention 骨架注释展示）
pub struct TransformerExpert {
    /// 领域 ID
    did: u32,
    /// 输入维度
    in_dim: usize,
    /// 输出维度
    out_dim: usize,
    /// 注意力头数（FlashAttention 必需，Phase 4 补全时用到）
    #[allow(dead_code)]
    num_heads: usize,
    // ===== 7 组参数（严格对齐设计文档 §九）=====
    /// LN1：注意力前归一化
    ln1: LayerNorm,
    /// QKV 投影：d_model → 3*d_model（拆 Q/K/V 各 [batch, heads, seq, d_k]）
    qkv: Linear,
    /// 注意力输出投影：d_model → d_model
    o_proj: Linear,
    /// LN2：FFN 前归一化
    ln2: LayerNorm,
    /// FFN1：d_model → 4*d_model（隐藏层扩张）
    ffn1: Linear,
    /// FFN2：4*d_model → d_model（投影回 d_model）
    ffn2: Linear,
    /// 最终输出投影：d_model → out_dim
    proj: Linear,
}

impl TransformerExpert {
    /// 构造 TransformerExpert（5 独立参数，对齐设计文档 §九 签名）
    ///
    /// ffn 隐藏层固定 4*d_model（Transformer 标准配置）
    pub fn new(did: u32, in_dim: usize, d_model: usize, out_dim: usize, num_heads: usize) -> Self {
        // d_model 必须能被 num_heads 整除（FlashAttention 要求每个 head 维度相同）
        assert_eq!(d_model % num_heads, 0, "d_model 必须能被 num_heads 整除");
        // FFN 隐藏层维度（Transformer 标准 4× 扩张）
        let d_ffn = 4 * d_model;
        Self {
            did, in_dim, out_dim, num_heads,
            // ===== 7 组参数（严格对齐设计文档 §九）=====
            ln1:   LayerNorm::new(did, "trans.ln1",  d_model),
            qkv:   Linear::new(did, "trans.qkv",    d_model, 3 * d_model),
            o_proj: Linear::new(did, "trans.o",     d_model, d_model),
            ln2:   LayerNorm::new(did, "trans.ln2",  d_model),
            ffn1:  Linear::new(did, "trans.ffn1",   d_model, d_ffn),
            ffn2:  Linear::new(did, "trans.ffn2",   d_ffn,   d_model),
            proj:  Linear::new(did, "trans.proj",   d_model, out_dim),
        }
    }
}

impl DomainExpert for TransformerExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_dim }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { oyta_tensor::is_training_mode() }

    /// forward：LN1 → QKV → FlashAttention(骨架) → o_proj → 残差 → LN2 → FFN1 → GELU → FFN2 → 残差 → proj
    ///
    /// FlashAttention 完整调用（骨架注释，等 oyta-tensor Backend trait 补全后启用）：
    /// ```ignore
    /// // 1. LN1 → QKV 投影
    /// let ln1_out = self.ln1.forward(x);
    /// let qkv = self.qkv.forward(&ln1_out);  // [batch, seq, 3*d_model]
    ///
    /// // 2. 拆分 Q/K/V + reshape 成 [batch, heads, seq, d_k]
    /// let d_k = d_model / num_heads;
    /// // q = qkv[:, :, 0:d_model].reshape([batch, seq, heads, d_k]).transpose(1, 2)  // [batch, heads, seq, d_k]
    /// // k = qkv[:, :, d_model:2*d_model].reshape(...).transpose(1, 2)
    /// // v = qkv[:, :, 2*d_model:].reshape(...).transpose(1, 2)
    ///
    /// // 3. FlashAttention 调用（oyta_ops::attention::flash::flash_attention_multihead）
    /// let scale = 1.0 / (d_k as f32).sqrt();
    /// let attn_out = flash_attention_multihead(&q, &k, &v, self.num_heads, false, scale);
    ///
    /// // 4. 输出投影 + 残差
    /// let out = self.o_proj.forward(&attn_out);
    /// let x = x.add(&out).expect("Transformer attn 残差失败");
    ///
    /// // 5. LN2 → FFN → GELU → FFN → 残差 → proj
    /// let ln2_out = self.ln2.forward(&x);
    /// let ffn1_out = self.ffn1.forward(&ln2_out).gelu().unwrap();
    /// let ffn2_out = self.ffn2.forward(&ffn1_out);
    /// let x = x.add(&ffn2_out).expect("Transformer FFN 残差失败");
    /// self.proj.forward(&x)
    /// ```
    ///
    /// Phase 3 简化实现：FlashAttention 链路暂跳过，直接 LN1 → LN2 → FFN → 残差 → proj
    /// 7 组参数全部初始化注册，框架层 parameter_pool 计数正确
    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // ===== 7 组参数结构完整注册，Phase 3 forward 简化路径 =====
        // Step 1: LN1（注意力前归一化）
        let ln1_out = self.ln1.forward(x);

        // Step 2: QKV + FlashAttention 骨架（Phase 4 补全）
        // let qkv = self.qkv.forward(&ln1_out);
        // let attn_out = flash_attention_multihead(...);
        // let o = self.o_proj.forward(&attn_out);
        // let x = x.add(&o).expect("attn 残差");
        // 当前简化：跳过 FlashAttention 链路，用 ln1_out 直接进入 LN2
        let attn_skip = self.o_proj.forward(&ln1_out); // 走 o_proj 保证参数参与 forward
        let after_attn = x.add(&attn_skip).expect("Transformer attn skip 残差失败");

        // Step 3: LN2（FFN 前归一化）
        let ln2_out = self.ln2.forward(&after_attn);
        // Step 4: FFN1 → GELU → FFN2
        let ffn1_out = self.ffn1.forward(&ln2_out);
        let ffn1_act = ffn1_out.gelu().expect("Transformer FFN1 gelu 失败");
        let ffn2_out = self.ffn2.forward(&ffn1_act);
        // Step 5: 残差连接
        let res = after_attn.add(&ffn2_out).expect("Transformer FFN 残差 add 失败");
        // Step 6: proj → 输出
        self.proj.forward(&res)
    }

    /// 7 组参数（严格对齐设计文档 §九）
    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.ln1.parameters());
        all.extend(self.qkv.parameters());
        all.extend(self.o_proj.parameters());
        all.extend(self.ln2.parameters());
        all.extend(self.ffn1.parameters());
        all.extend(self.ffn2.parameters());
        all.extend(self.proj.parameters());
        all
    }
}
