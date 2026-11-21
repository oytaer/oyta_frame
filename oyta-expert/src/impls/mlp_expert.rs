//! MLPExpert：Linear → LayerNorm → GELU → Linear
//!
//! 全部底层算子调用 oyta_tensor API：
//!   Linear::forward → Tensor::matmul + Tensor::add
//!   LayerNorm::forward → Tensor::layernorm
//!   Tensor::gelu（直接调 oyta_tensor）
//!   Tensor::transpose（Linear 内部转置 W）

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::traits::domain_expert::DomainExpert;
use crate::utils::linear::Linear;
use crate::utils::norm::LayerNorm;

/// MLP 专家（典型 Transformer FFN 结构）
pub struct MLPExpert {
    /// 该专家绑定的领域 ID
    did: u32,
    /// 输入维度
    in_dim: usize,
    /// 输出维度
    out_dim: usize,
    /// 隐藏层：in → hidden
    hidden: Linear,
    /// LayerNorm（归一化）
    ln: LayerNorm,
    /// 输出层：hidden → out
    out: Linear,
}

impl MLPExpert {
    /// 构造 MLP 专家
    pub fn new(did: u32, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            did,
            in_dim,
            out_dim,
            hidden: Linear::new(did, "mlp.hidden", in_dim, hidden_dim),
            ln:     LayerNorm::new(did, "mlp.ln",    hidden_dim),
            out:    Linear::new(did, "mlp.out",       hidden_dim, out_dim),
        }
    }
}

impl DomainExpert for MLPExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_dim }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { oyta_tensor::is_training_mode() }

    /// forward 全链路：Linear → LN → GELU → Linear
    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // Step 1: hidden Linear → [batch, hidden_dim]
        let h1 = self.hidden.forward(x);
        // Step 2: LayerNorm 归一化
        let h2 = self.ln.forward(&h1);
        // Step 3: GELU 激活（直接调 oyta_tensor::Tensor::gelu）
        let h3 = h2.gelu().expect("MLPExpert::forward gelu 失败");
        // Step 4: out Linear → [batch, out_dim]
        self.out.forward(&h3)
    }

    /// 收集所有可训练参数（hidden.weight+bias + ln.gamma+beta + out.weight+bias）
    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.hidden.parameters());
        all.extend(self.ln.parameters());
        all.extend(self.out.parameters());
        all
    }
}
