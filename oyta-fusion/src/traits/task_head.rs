//! TaskHeadBase trait：所有任务头的统一接口

use oyta_tensor::prelude::*;
use oyta_router::Parameter;

/// 任务输出头统一接口
pub trait TaskHeadBase: Send + Sync {
    /// forward：fused 特征 → logits（Classification）或标量（Regression）
    fn forward(&self, fused: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;
    /// 该任务头的所有可训练参数
    fn parameters(&self) -> Vec<Parameter>;
    /// 输出维度（分类=num_classes，回归=1）
    fn output_dim(&self) -> usize;
}
