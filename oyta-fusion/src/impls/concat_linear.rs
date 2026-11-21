//! ConcatLinearFusion：concat 激活专家输出 → Linear 投影
//!
//! Linear 内部调 oyta_tensor（matmul + add）

use oyta_tensor::prelude::*;
use oyta_expert::Linear;
use oyta_router::Parameter;
use crate::traits::fusion_strategy::FusionStrategyTrait;
use crate::utils::tensor_concat::concat_tensors;

/// 拼接 + Linear 投影融合策略
pub struct ConcatLinearFusion {
    /// 专家输出维度（每个专家输出的特征维）
    expert_dim: usize,
    /// 最大激活专家数（concat 时 padding 到这个数）
    max_active: usize,
    /// 投影后融合维度
    proj_dim: usize,
    /// Linear 投影层（复用 oyta_expert::Linear）
    linear: Linear,
}

impl ConcatLinearFusion {
    pub fn new(did: u32, expert_dim: usize, proj_dim: usize) -> Self {
        // MoE 典型配置：Top-4 专家
        let max_active = 4;
        Self {
            expert_dim, max_active, proj_dim,
            linear: Linear::new(did, "fusion.proj", max_active * expert_dim, proj_dim),
        }
    }
}

impl FusionStrategyTrait for ConcatLinearFusion {
    fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], _weights: &[f32]) -> Tensor<f32, Cpu> {
        // 空集兜底
        if outputs.is_empty() {
            return Tensor::<f32, Cpu>::zeros([1, self.proj_dim]);
        }

        let batch = outputs[0].1.shape().dims[0];
        let num_active = outputs.len();

        // Step 1: 先按 dim=1 拼接所有激活专家输出
        let refs: Vec<&Tensor<f32, Cpu>> = outputs.iter().map(|(_, t)| t).collect();
        let concatted = concat_tensors(&refs, 1); // [batch, num_active * expert_dim]

        // Step 2: padding 到 max_active（num_active < max_active 时补零）
        let current_concat_dim = num_active * self.expert_dim;
        let target_concat_dim = self.max_active * self.expert_dim;
        let padded = if current_concat_dim < target_concat_dim {
            let mut pd = concatted.data_slice().to_vec();
            pd.extend(std::iter::repeat(0.0f32).take(batch * (target_concat_dim - current_concat_dim)));
            Tensor::<f32, Cpu>::from_data(pd, [batch, target_concat_dim])
                .expect("ConcatLinear padding 构造失败")
        } else {
            concatted
        };

        // Step 3: Linear 投影（内部调 oyta_tensor matmul + add）
        self.linear.forward(&padded)
    }

    fn fused_dim(&self) -> usize { self.proj_dim }

    fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
}
