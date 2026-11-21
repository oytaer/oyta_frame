//! SparseMaskSet：三层稀疏掩码全集
//!
//! L1 领域级（domain_mask: Vec<bool>）           —— 整个专家激活/跳过
//! L2 通道级（channel_masks: HashMap<u32, Vec<f32>>） —— 领域内特征选择/Early Exit
//! L3 序列级（sequence_masks: HashMap<(u32, usize), Vec<f32>>） —— 变长/padding

use std::collections::HashMap;

/// 稀疏掩码层级枚举（由粗到细，每个可以独立开关）
#[derive(Debug, Clone)]
pub enum MaskLevel {
    /// Level 1: 领域级（整个专家激活/跳过）
    DomainMask,
    /// Level 2: 通道级（领域内特征选择/Early Exit）
    ChannelMask {
        expert_domain_id: u32,
    },
    /// Level 3: 序列级（变长/padding）
    SequenceMask {
        expert_domain_id: u32,
        channel: usize,
    },
}

/// 单个领域的 Op 消费掩码（给 autograd 层 resolve_op_mask 用）
#[derive(Debug, Clone, Default)]
pub struct DomainSparseMask {
    /// L2 通道级 mask（mask[f] ∈ [0,1]，0=参与, 1=mask 掉）
    pub channel_mask: Option<Vec<f32>>,
    /// L3 序列级 mask（mask[t] ∈ [0,1]，0=有效, 1=padding）
    pub sequence_mask: Option<Vec<f32>>,
}

/// 完整三层掩码集（一个 batch 的全部稀疏决策）
#[derive(Debug, Clone)]
pub struct SparseMaskSet {
    /// L1: 领域级 mask，len=num_domains，true=激活
    pub domain_mask: Vec<bool>,
    /// L2: 通道级 mask，domain_id → mask Vec（mask[f] ∈ [0,1]）
    pub channel_masks: HashMap<u32, Vec<f32>>,
    /// L3: 序列级 mask，(domain_id, channel) → mask Vec（mask[t] ∈ [0,1]）
    pub sequence_masks: HashMap<(u32, usize), Vec<f32>>,
}

impl SparseMaskSet {
    /// 构造：所有领域默认未激活
    pub fn new(num_domains: usize) -> Self {
        Self {
            domain_mask: vec![false; num_domains],
            channel_masks: HashMap::new(),
            sequence_masks: HashMap::new(),
        }
    }

    // ===== L1 领域级 =====

    /// 激活/冻结某个领域（true=激活）
    pub fn set_domain_active(&mut self, did: u32, active: bool) {
        let i = did as usize;
        if i < self.domain_mask.len() {
            self.domain_mask[i] = active;
        }
    }

    /// 查询领域是否激活
    pub fn is_active(&self, did: u32) -> bool {
        self.domain_mask.get(did as usize).copied().unwrap_or(false)
    }

    /// 返回所有激活领域 ID（按 domain_id 升序）
    pub fn active_ids(&self) -> Vec<u32> {
        self.domain_mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m)
            .map(|(i, _)| i as u32)
            .collect()
    }

    // ===== L2 通道级 =====

    /// 设置某个领域的通道级 mask
    pub fn set_channel_mask(&mut self, did: u32, mask: Vec<f32>) {
        self.channel_masks.insert(did, mask);
    }

    /// 查询某个领域的通道级 mask（可选）
    pub fn get_channel_mask(&self, did: u32) -> Option<&Vec<f32>> {
        self.channel_masks.get(&did)
    }

    // ===== L3 序列级 =====

    /// 设置某个领域某个通道的序列级 mask
    pub fn set_sequence_mask(&mut self, did: u32, channel: usize, mask: Vec<f32>) {
        self.sequence_masks.insert((did, channel), mask);
    }

    /// 查询某个领域默认通道(0)的序列级 mask
    pub fn get_sequence_mask(&self, did: u32) -> Option<&Vec<f32>> {
        self.sequence_masks.get(&(did, 0))
    }

    /// 查询某个领域某个通道的序列级 mask（autograd Op 消费 mask 用）
    ///
    /// 约定：如果 (did, channel) 没找到 → fallback 查 (did, 0)
    /// 因为卷积层所有输出通道共享 padding mask，router 层只注入 channel=0
    /// 这样 build() 不用硬编码 num_channels，autograd 层调这个 API 统一 fallback
    pub fn get_sequence_mask_for_op(&self, did: u32, channel: usize) -> Option<&Vec<f32>> {
        self.sequence_masks.get(&(did, channel))
            .or_else(|| self.sequence_masks.get(&(did, 0)))
    }

    // ===== 工厂方法（完整三层生成流程）=====

    /// 从 ContentAnalysis.domain_weights 生成完整三层掩码集合
    ///
    /// 完整流程（架构 0.6.8）：
    ///   1. RoutingStrategy.generate_domain_mask → L1 domain_mask
    ///      TopK: 降序排 → 前 K 个；Threshold: 权重 > 阈值
    ///      保证至少激活一个（空集 fallback）
    ///   2. 注入预设 L2 channel_mask（只给激活专家）
    ///   3. 注入 ContentAnalyzer 检测到的 L3 padding_mask（只给激活专家）
    pub fn build(
        weights: &[(u32, f32)],
        num_domains: usize,
        strategy: &crate::impls::routing_strategy::RoutingStrategy,
        preset_channels: &HashMap<u32, Option<Vec<f32>>>,
        preset_sequences: &HashMap<u32, Option<Vec<f32>>>,
    ) -> Self {
        // Step 1: 生成 L1 domain_mask
        let mut mask_set = SparseMaskSet::new(num_domains);
        let dm = strategy.generate_domain_mask(weights, num_domains);
        for (i, &active) in dm.iter().enumerate() {
            mask_set.set_domain_active(i as u32, active);
        }

        // Step 2: 注入 L2 channel_mask（只给激活专家）
        for (did, mask_opt) in preset_channels {
            if mask_set.is_active(*did) {
                if let Some(mask) = mask_opt {
                    mask_set.set_channel_mask(*did, mask.clone());
                }
            }
        }

        // Step 3: 注入 L3 sequence_mask（只给激活专家，卷积层所有输出通道共享 padding mask）
        for (did, mask_opt) in preset_sequences {
            if mask_set.is_active(*did) {
                if let Some(mask) = mask_opt {
                    // 通道 0 注入（DomainRouter 不硬编码 num_channels，autograd 层遍历全部通道）
                    mask_set.set_sequence_mask(*did, 0, mask.clone());
                }
            }
        }

        mask_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impls::routing_strategy::RoutingStrategy;

    #[test]
    fn test_domain_mask_basic() {
        let mut ms = SparseMaskSet::new(4);
        ms.set_domain_active(1, true);
        ms.set_domain_active(3, true);
        assert_eq!(ms.active_ids(), vec![1, 3]);
        assert!(ms.is_active(1));
        assert!(!ms.is_active(0));
    }

    #[test]
    fn test_build_topk() {
        let weights = vec![(0, 0.05), (1, 0.85), (2, 0.03), (3, 0.07)];
        let strategy = RoutingStrategy::top_k(2);
        let preset_channels: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let preset_sequences: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let ms = SparseMaskSet::build(&weights, 4, &strategy, &preset_channels, &preset_sequences);
        // 降序: (1,0.85), (3,0.07), (0,0.05), (2,0.03) → Top2 = [1,3]
        assert_eq!(ms.active_ids(), vec![1, 3]);
    }

    #[test]
    fn test_build_threshold() {
        let weights = vec![(0, 0.05), (1, 0.85), (2, 0.03), (3, 0.07)];
        let strategy = RoutingStrategy::threshold(0.1);
        let preset_channels: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let preset_sequences: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let ms = SparseMaskSet::build(&weights, 4, &strategy, &preset_channels, &preset_sequences);
        // 只有 did=1 权重 0.85 > 0.1
        assert_eq!(ms.active_ids(), vec![1]);
    }

    #[test]
    fn test_empty_fallback_activates_highest() {
        // 所有权重=0 → fallback 激活权重最高的那个（这里是 0.1）
        let weights = vec![(0, 0.0), (1, 0.0), (2, 0.1), (3, 0.0)];
        let strategy = RoutingStrategy::threshold(0.5);
        let preset_channels: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let preset_sequences: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
        let ms = SparseMaskSet::build(&weights, 4, &strategy, &preset_channels, &preset_sequences);
        assert!(!ms.active_ids().is_empty(), "空集 fallback 应至少激活一个领域");
    }

    /// get_sequence_mask_for_op：查不到特定通道时 fallback 到 channel=0
    #[test]
    fn test_get_sequence_mask_fallback() {
        let mut ms = SparseMaskSet::new(2);
        ms.set_domain_active(0, true);

        // 只注入 channel=0 的 sequence mask
        let mask = vec![0.0f32, 0.0, 1.0, 1.0];
        ms.set_sequence_mask(0, 0, mask.clone());

        // 直接查 channel=0 → 命中
        assert_eq!(ms.get_sequence_mask_for_op(0, 0), Some(&mask));

        // 查 channel=5 → 自动 fallback 到 channel=0
        assert_eq!(ms.get_sequence_mask_for_op(0, 5), Some(&mask));

        // 不存在的 domain → None
        assert_eq!(ms.get_sequence_mask_for_op(99, 0), None);

        // 不存在的 domain + 任意 channel → None
        assert_eq!(ms.get_sequence_mask_for_op(99, 5), None);
    }
}
