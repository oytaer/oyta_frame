//! FusionStrategyTrait：所有融合策略的统一接口
//!
//! 框架层 FeatureFusion 通过这个 trait 分发到具体实现

use oyta_tensor::prelude::*;
use oyta_router::Parameter;

/// 融合策略统一接口
pub trait FusionStrategyTrait: Send + Sync {
    /// 融合激活专家输出 + Router 归一化权重 → fused 特征
    fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu>;

    /// 该融合策略的输出维度（FeatureFusion.fused_dim 用）
    fn fused_dim(&self) -> usize;

    /// 该融合策略的所有可训练参数（WeightedSum 返回空 Vec）
    fn parameters(&self) -> Vec<Parameter>;
}
