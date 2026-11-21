//! FeatureFusion 主类：策略分发 + 参数收集 + train/eval
//!
//! 框架层融合入口：根据 FusionStrategy 分发到 WeightedSum / ConcatLinear / CrossAttention
//! 所有可训练参数自动收集（WeightedSum 无可训练参数）

use oyta_tensor::prelude::*;
use oyta_router::Parameter;
use crate::types::fusion_config::FusionStrategy;
use crate::traits::fusion_strategy::FusionStrategyTrait;
use crate::impls::weighted_sum::WeightedSumFusion;
use crate::impls::concat_linear::ConcatLinearFusion;
use crate::impls::cross_attention::CrossAttentionFusion;

/// 特征融合主类（策略分发 + 统一接口）
pub struct FeatureFusion {
    /// 融合策略枚举
    strategy: FusionStrategy,
    /// 融合后输出维度
    fused_dim: usize,
    // ===== 三选一指针对应的实现（其他为 None）=====
    weighted_sum: Option<WeightedSumFusion>,
    concat_linear: Option<ConcatLinearFusion>,
    cross_attention: Option<CrossAttentionFusion>,
    /// 领域 ID（注册参数用）
    did: u32,
    /// 当前训练态（影响 Dropout + BatchNorm）
    training: bool,
}

impl FeatureFusion {
    /// 构造 FeatureFusion
    ///
    /// expert_dim 是专家输出的特征维（三种策略都需要）
    pub fn new(strategy: FusionStrategy, did: u32, expert_dim: usize) -> Self {
        // 根据策略计算 fused_dim
        let fused_dim = match &strategy {
            FusionStrategy::WeightedSum => expert_dim,
            FusionStrategy::ConcatLinear { proj_dim } => *proj_dim,
            FusionStrategy::CrossAttention { proj_dim, .. } => *proj_dim,
        };
        Self {
            fused_dim,
            strategy: strategy.clone(),
            did,
            // 根据策略初始化对应实现
            weighted_sum: matches!(strategy, FusionStrategy::WeightedSum)
                .then(|| WeightedSumFusion::new(expert_dim)),
            concat_linear: if let FusionStrategy::ConcatLinear { proj_dim } = &strategy {
                Some(ConcatLinearFusion::new(did, expert_dim, *proj_dim))
            } else { None },
            cross_attention: if let FusionStrategy::CrossAttention { num_heads, dropout, proj_dim } = &strategy {
                Some(CrossAttentionFusion::new(did, expert_dim, *num_heads, *dropout, *proj_dim))
            } else { None },
            training: true, // 默认训练态
        }
    }

    /// 融合主入口：根据 strategy 分发到具体实现
    pub fn fuse(&self, outputs: &[(u32, Tensor<f32, Cpu>)], weights: &[f32]) -> Tensor<f32, Cpu> {
        match &self.strategy {
            FusionStrategy::WeightedSum => {
                self.weighted_sum.as_ref().unwrap().fuse(outputs, weights)
            }
            FusionStrategy::ConcatLinear { .. } => {
                self.concat_linear.as_ref().unwrap().fuse(outputs, weights)
            }
            FusionStrategy::CrossAttention { .. } => {
                self.cross_attention.as_ref().unwrap().fuse(outputs, weights)
            }
        }
    }

    /// 收集所有可训练参数（WeightedSum 返回空 Vec）
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut all = Vec::new();
        if let Some(ws) = &self.weighted_sum { all.extend(ws.parameters()); }
        if let Some(cl) = &self.concat_linear { all.extend(cl.parameters()); }
        if let Some(ca) = &self.cross_attention { all.extend(ca.parameters()); }
        all
    }

    pub fn fused_dim(&self) -> usize { self.fused_dim }

    pub fn did(&self) -> u32 { self.did }

    /// 切换训练态（影响内部模块 Dropout / BatchNorm）
    pub fn train(&mut self) {
        self.training = true;
        oyta_tensor::set_training_mode(true);
    }

    /// 切换推理态
    pub fn eval(&mut self) {
        self.training = false;
        oyta_tensor::set_training_mode(false);
    }

    pub fn is_training(&self) -> bool { self.training }
}
