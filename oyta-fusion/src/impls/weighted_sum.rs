//! WeightedSumFusion：Σ gating_weight[i] × expert_output[i]
//!
//! 无可训练参数，纯 Vec 逐元素累加

use oyta_tensor::prelude::*;
use crate::traits::fusion_strategy::FusionStrategyTrait;
use oyta_router::Parameter;

/// 加权求和融合策略
pub struct WeightedSumFusion {
    /// 专家输出维度（所有专家输出必须相同）
    expert_dim: usize,
}

impl WeightedSumFusion {
    pub fn new(expert_dim: usize) -> Self {
        Self { expert_dim }
    }
}

impl FusionStrategyTrait for WeightedSumFusion {
    fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu> {
        // 空集兜底：返回零 Tensor
        if outputs.is_empty() {
            return Tensor::<f32, Cpu>::zeros([1, self.expert_dim]);
        }
        assert_eq!(weights.len(), outputs.len(), "WeightedSum weights 数量必须等于 outputs");

        // 取第一个输出的 shape 作为融合后 shape
        let first_shape = outputs[0].1.shape().dims.clone();
        let total = first_shape.iter().product::<usize>();
        // 逐元素累加：fused[j] = Σ weights[i] * expert_out[i][j]
        let mut fused = vec![0.0f32; total];
        for (i, (_, expert_out)) in outputs.iter().enumerate() {
            let w = weights[i];
            let d = expert_out.data_slice();
            for j in 0..total {
                fused[j] += w * d[j];
            }
        }
        Tensor::<f32, Cpu>::from_data(fused, first_shape)
            .expect("WeightedSum fused 构造失败")
    }

    fn fused_dim(&self) -> usize { self.expert_dim }

    fn parameters(&self) -> Vec<Parameter> { Vec::new() }
}
