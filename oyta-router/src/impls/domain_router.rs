//! DomainRouter：组装 RoutingStrategy + SparseMaskSet + SparseParameterPool
//!
//! 对外暴露 route() 方法（每个 batch 动态决策）
//! 一个 DomainRouter 实例绑定：strategy + num_domains + preset_masks + param_pool

use std::collections::HashMap;
use crate::impls::routing_strategy::{RoutingStrategy, RoutingResult};
use crate::types::sparse_mask::SparseMaskSet;
use crate::utils::param_pool::SparseParameterPool;

pub struct DomainRouter {
    /// 稀疏路由策略（TopK / Threshold）
    strategy: RoutingStrategy,
    /// 总领域数（domain_mask 的长度）
    num_domains: usize,
    /// 预设 L2 channel_mask（框架初始化时配置，训练时不变）
    preset_channels: HashMap<u32, Option<Vec<f32>>>,
    /// 预设 L3 sequence_mask（框架初始化时配置，训练时可被 ContentAnalysis.padding_mask 覆盖）
    preset_sequences: HashMap<u32, Option<Vec<f32>>>,
    /// 按领域独立存储的参数池（专家模块注册参数用）
    pub param_pool: SparseParameterPool,
}

impl DomainRouter {
    /// 构造：指定路由策略 + 总领域数
    pub fn new(strategy: RoutingStrategy, num_domains: usize) -> Self {
        Self {
            strategy,
            num_domains,
            preset_channels: HashMap::new(),
            preset_sequences: HashMap::new(),
            param_pool: SparseParameterPool::new(),
        }
    }

    /// 注入某个领域的预设通道级 mask（框架初始化时配置 Early Exit）
    pub fn with_channel_mask(mut self, did: u32, mask: Vec<f32>) -> Self {
        self.preset_channels.insert(did, Some(mask));
        self
    }

    /// 注入某个领域的预设序列级 mask（框架初始化时配置固定长度序列 mask）
    pub fn with_sequence_mask(mut self, did: u32, mask: Vec<f32>) -> Self {
        self.preset_sequences.insert(did, Some(mask));
        self
    }

    pub fn num_domains(&self) -> usize {
        self.num_domains
    }

    /// 核心方法：一个 batch 的完整稀疏决策
    ///
    /// 输入: ContentAnalysis（来自 oyta-content）
    ///   - domain_weights: [(domain_id, softmax 权重)]
    ///   - padding_mask: 可选变长 mask（注入 L3 sequence_mask）
    ///
    /// 输出: (RoutingResult, SparseMaskSet)
    ///   - RoutingResult: active_indices（FeatureFusion 用）+ normalized_weights（gating 用）
    ///   - SparseMaskSet: 三层掩码全集（ExpertExecutor + sparse_backward 用）
    pub fn route(&self, analysis: &oyta_content::ContentAnalysis) -> (RoutingResult, SparseMaskSet) {
        let weights = &analysis.domain_weights;

        // Step 1: RoutingStrategy 生成决策（TopK/Threshold → active_indices + softmax 权重）
        let result = self.strategy.route(weights, self.num_domains);

        // Step 2: 拼接预设 L3 sequence_mask + ContentAnalyzer 检测到的 padding_mask
        // ContentAnalyzer 检测到的 padding_mask 是 batch 内第一个样本的 mask
        // 如果存在，给所有激活领域注入
        let mut seq_input = self.preset_sequences.clone();
        if let Some(ref pm) = analysis.padding_mask {
            if let Some(first) = pm.first() {
                for did in &result.active_indices {
                    seq_input.insert(*did, Some(first.clone()));
                }
            }
        }

        // Step 3: SparseMaskSet::build 组装三层掩码全集
        let mask_set = SparseMaskSet::build(
            weights,
            self.num_domains,
            &self.strategy,
            &self.preset_channels,
            &seq_input,
        );

        (result, mask_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impls::routing_strategy::RoutingStrategy;
    use oyta_content::ContentAnalysis;
    use oyta_tensor::{Tensor, Cpu};

    /// 构造一个测试用 ContentAnalysis（模拟 oyta-content 输出）
    fn make_analysis(weights: Vec<(u32, f32)>) -> ContentAnalysis {
        let fv = Tensor::<f32, Cpu>::randn([1, 256]);
        ContentAnalysis {
            domain_weights: weights,
            feature_vector: fv,
            evidence: Vec::new(),
            padding_mask: None,
        }
    }

    /// DomainRouter TopK(2) 对 weights=[(0,0.05),(1,0.85),(2,0.03),(3,0.07)] → active=[1,3]
    #[test]
    fn test_router_topk_route() {
        let router = DomainRouter::new(RoutingStrategy::top_k(2), 4);
        let analysis = make_analysis(vec![(0, 0.05), (1, 0.85), (2, 0.03), (3, 0.07)]);
        let (result, mask_set) = router.route(&analysis);
        assert_eq!(result.active_indices, vec![1, 3]);
        assert_eq!(mask_set.active_ids(), vec![1, 3]);
        assert!(!mask_set.is_active(0));
        assert!(!mask_set.is_active(2));
    }

    /// Threshold: 所有权重=0.01 → fallback 激活至少一个
    #[test]
    fn test_router_empty_fallback() {
        let router = DomainRouter::new(RoutingStrategy::threshold(0.5), 4);
        let analysis = make_analysis(vec![(0, 0.0), (1, 0.1), (2, 0.0), (3, 0.0)]);
        let (result, mask_set) = router.route(&analysis);
        assert!(!result.active_indices.is_empty(), "fallback 应至少激活一个");
        assert!(!mask_set.active_ids().is_empty());
    }
}
