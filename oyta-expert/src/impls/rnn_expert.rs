//! RNNExpert 骨架
//!
//! RNN 专家涉及循环状态（hidden/cell），完整实现需要：
//!   - LSTM/GRU 单元算子（oyta-tensor 暂无，需要 Backend trait 扩展）
//!   - 序列级 unroll（scan）处理变长序列
//!
//! Phase 3 先放骨架结构，forward 暂时用 Linear 占位
//! 核心价值是展示 DomainExpert trait 的实现方式 + 框架层正确装配

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::traits::domain_expert::DomainExpert;
use crate::utils::linear::Linear;

/// RNN 专家（骨架）
pub struct RNNExpert {
    did: u32,
    in_dim: usize,
    out_dim: usize,
    /// 隐藏层占位（实际应是 LSTM/GRU）
    hidden: Linear,
    /// 输出投影
    proj: Linear,
}

impl RNNExpert {
    pub fn new(did: u32, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            did, in_dim, out_dim,
            hidden: Linear::new(did, "rnn.hidden", in_dim, hidden_dim),
            proj:   Linear::new(did, "rnn.proj",   hidden_dim, out_dim),
        }
    }
}

impl DomainExpert for RNNExpert {
    fn domain_id(&self) -> u32 { self.did }
    fn input_dim(&self) -> usize { self.in_dim }
    fn output_dim(&self) -> usize { self.out_dim }
    fn is_training(&self) -> bool { oyta_tensor::is_training_mode() }

    /// 骨架实现：hidden → 底层 tanh → proj
    fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let h = self.hidden.forward(x);
        // 底层 tanh：Tensor.tanh() → D::elwise_tanh Rayon 并行（返回 Self，内部 expect）
        let tanh_h = h.tanh();
        self.proj.forward(&tanh_h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        all.extend(self.hidden.parameters());
        all.extend(self.proj.parameters());
        all
    }
}
