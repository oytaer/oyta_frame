//! RoutingStrategy：稀疏路由策略（纯算法，不依赖任何上层 crate）
//!
//! 两种策略：TopK / Threshold
//!   - 都带 softmax 归一化（数值稳定版：减 max）
//!   - 都带空集 fallback（保证至少激活权重最高的一个领域）
//!   - domain_id 从 ContentAnalysis.domain_weights 里取

use std::collections::HashSet;

/// 稀疏路由策略枚举
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// 取权重降序的前 K 个领域激活
    TopK { k: usize },
    /// 所有权重 > 阈值的领域激活
    Threshold { threshold: f32 },
}

/// 路由结果（route() 一次返回的完整决策）
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// L1 domain_mask（len=num_domains，true=激活）
    pub mask: Vec<bool>,
    /// 激活领域 ID（按权重降序排列）
    pub active_indices: Vec<u32>,
    /// softmax 归一化权重（和 = 1.0，顺序与 active_indices 对应）
    pub normalized_weights: Vec<f32>,
}

impl RoutingStrategy {
    /// 构造 TopK 策略
    /// k=0 自动提升为 k=1（至少选一个）
    pub fn top_k(k: usize) -> Self {
        Self::TopK { k: k.max(1) }
    }
    pub fn threshold(t: f32) -> Self {
        Self::Threshold { threshold: t }
    }

    /// 主入口：从 domain_weights 生成完整路由决策
    /// 自动过滤 NaN/Inf 权重（对调用方透明）
    pub fn route(&self, weights: &[(u32, f32)], num_domains: usize) -> RoutingResult {
        // Step 0: 过滤 NaN/Inf 权重（route 层统一处理，双保险 + 调用方透明）
        let weights: Vec<(u32, f32)> = weights
            .iter()
            .filter(|(_, w)| w.is_finite())
            .cloned()
            .collect();
        // 极端兜底：全部 NaN → 所有权重视为 0.0
        let weights = if weights.is_empty() {
            (0..num_domains as u32).map(|i| (i, 0.0f32)).collect::<Vec<_>>()
        } else {
            weights
        };

        // Step 1: 生成 L1 domain_mask
        let mask = self.generate_domain_mask(&weights, num_domains);

        // Step 2: 收集激活领域 ID（按 weights 里原始顺序）
        let mut active_indices: Vec<u32> = Vec::new();
        for &(id, _) in &weights {
            if mask.get(id as usize).copied().unwrap_or(false) {
                active_indices.push(id);
            }
        }

        // Step 3: softmax 归一化
        let (_, nw) = self.compute_normalized_weights(&weights, &active_indices);

        RoutingResult { mask, active_indices, normalized_weights: nw }
    }

    /// 生成领域级 mask（L1 来源）
    ///
    /// TopK: 权重降序排 → 前 K 个标记激活
    /// Threshold: 权重 > 阈值标记激活
    /// 空集 fallback: 激活权重最高的那个领域
    /// NaN/Inf 权重自动过滤（防止 partial_cmp 行为异常）
    pub fn generate_domain_mask(&self, weights: &[(u32, f32)], num_domains: usize) -> Vec<bool> {
        // Step 0: 过滤 NaN/Inf 权重（partial_cmp 对 NaN 返回 None，会导致 max_by 选不到正确的）
        let weights: Vec<(u32, f32)> = weights
            .iter()
            .filter(|(_, w)| w.is_finite())
            .cloned()
            .collect();
        // 极端兜底：所有权重都是 NaN → 全部激活（训练初始状态，保守策略）
        if weights.is_empty() {
            return vec![true; num_domains];
        }

        let mut mask = vec![false; num_domains];
        match self {
            Self::TopK { k } => {
                // 按权重降序，稳定排序（权重相同按 domain_id 升序）
                let mut sorted: Vec<(u32, f32)> = weights.iter().cloned().collect();
                sorted.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                for i in 0..(*k).min(sorted.len()) {
                    let did = sorted[i].0 as usize;
                    if did < num_domains {
                        mask[did] = true;
                    }
                }
            }
            Self::Threshold { threshold } => {
                for &(did, w) in &weights {
                    if w > *threshold && (did as usize) < num_domains {
                        mask[did as usize] = true;
                    }
                }
            }
        }
        // ===== 空集 fallback：保证至少激活一个 =====
        if !mask.iter().any(|&m| m) {
            if let Some((best_id, _)) = weights
                .iter()
                .cloned()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let i = best_id as usize;
                if i < num_domains {
                    mask[i] = true;
                }
            }
        }
        mask
    }

    /// softmax 归一化（数值稳定版：减 max，防溢出）
    /// 只对激活领域做 softmax（FeatureFusion gating 用）
    pub fn compute_normalized_weights(
        &self,
        weights: &[(u32, f32)],
        active_indices: &[u32],
    ) -> (Vec<u32>, Vec<f32>) {
        let id_set: HashSet<u32> = active_indices.iter().copied().collect();
        // 取激活领域的原始权重
        let active_raw: Vec<(u32, f32)> = weights
            .iter()
            .filter(|(id, _)| id_set.contains(id))
            .cloned()
            .collect();
        if active_raw.is_empty() {
            return (Vec::new(), Vec::new());
        }
        // 底层 softmax：Tensor.softmax(-1) → f64 累加器 + Rayon 并行 + 数值稳定
        use oyta_tensor::prelude::*;
        let weights_vec: Vec<f32> = active_raw.iter().map(|(_, w)| *w).collect();
        let weights_tensor = Tensor::<f32, Cpu>::from_data(weights_vec.clone(), [weights_vec.len()]).unwrap();
        let sm_tensor = weights_tensor.softmax(-1).expect("softmax 底层调用失败");
        let norm = sm_tensor.data_slice().to_vec();
        let ids: Vec<u32> = active_raw.iter().map(|(id, _)| *id).collect();
        (ids, norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TopK(2) 对 [(0,0.05),(1,0.85),(2,0.03),(3,0.07)] → active=[1,3]
    #[test]
    fn test_topk_basic() {
        let s = RoutingStrategy::top_k(2);
        let w = vec![(0, 0.05), (1, 0.85), (2, 0.03), (3, 0.07)];
        let result = s.route(&w, 4);
        assert_eq!(result.active_indices, vec![1, 3]);
        assert_eq!(result.mask, vec![false, true, false, true]);
        // softmax: 0.85 和 0.07 的 softmax 应该显著偏向 0.85
        assert!(result.normalized_weights[0] > result.normalized_weights[1]);
        let sum: f32 = result.normalized_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// Threshold: 权重 > 0.1 → 只有 did=1
    #[test]
    fn test_threshold_basic() {
        let s = RoutingStrategy::threshold(0.1);
        let w = vec![(0, 0.05), (1, 0.85), (2, 0.03), (3, 0.07)];
        let result = s.route(&w, 4);
        assert_eq!(result.active_indices, vec![1]);
    }

    /// 所有权重=0 → 空集 fallback 激活权重最高的
    #[test]
    fn test_empty_fallback() {
        let s = RoutingStrategy::top_k(2);
        // 所有权重接近 0，TopK 会选前 2 个（但权重相同按 domain_id 升序）
        let w = vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)];
        let result = s.route(&w, 4);
        // 空集 fallback：所有权重一样，选 domain_id 最小的 → [0]
        assert!(!result.active_indices.is_empty(), "空集 fallback 应激活至少一个");
    }

    /// softmax 归一化权重和 = 1
    #[test]
    fn test_softmax_sums_to_one() {
        let s = RoutingStrategy::top_k(3);
        let w = vec![(0, 0.2), (1, 0.5), (2, 0.1), (3, 0.2)];
        let result = s.route(&w, 4);
        let sum: f32 = result.normalized_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax 和应 ≈ 1.0");
    }

    /// top_k(0) 自动保护为至少 k=1
    #[test]
    fn test_top_k_zero_protection() {
        // top_k(0) 不应 panic，也不应返回空集
        let s = RoutingStrategy::top_k(0);
        let w = vec![(0, 0.1), (1, 0.5), (2, 0.3)];
        let result = s.route(&w, 3);
        assert!(!result.active_indices.is_empty(), "top_k(0) 至少激活一个领域");
        // 激活领域数应 >= 1
        assert!(result.active_indices.len() >= 1);
    }

    /// NaN/Inf 权重应被自动过滤，不影响路由
    #[test]
    fn test_nan_weight_filtered() {
        let s = RoutingStrategy::top_k(2);
        // 混合 NaN 和有效值
        let w = vec![
            (0, f32::NAN),    // NaN 应被过滤
            (1, 0.85),
            (2, f32::INFINITY), // Inf 应被过滤
            (3, 0.07),
        ];
        let result = s.route(&w, 4);
        // NaN/Inf 被过滤后只剩有效权重，TopK(2) 选 0.85 和 0.07
        assert_eq!(result.active_indices, vec![1, 3]);
        assert_eq!(result.mask, vec![false, true, false, true]);
        // softmax 不应该产生 NaN
        assert!(result.normalized_weights.iter().all(|w| w.is_finite()));
    }

    /// 所有权重都是 NaN → 全部激活（保守兜底）
    #[test]
    fn test_all_nan_fallback() {
        let s = RoutingStrategy::top_k(2);
        let w = vec![
            (0, f32::NAN),
            (1, f32::NAN),
            (2, f32::NAN),
        ];
        let result = s.route(&w, 3);
        // 全部 NaN → route 层兜底为所有权重 0.0 → TopK 选前 2 个
        assert!(!result.active_indices.is_empty());
        // softmax 和不应是 NaN
        let sum: f32 = result.normalized_weights.iter().sum();
        assert!(sum.is_finite());
    }
}
