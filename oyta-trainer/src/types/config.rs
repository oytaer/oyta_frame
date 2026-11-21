//! SparsityConfig + TrainerConfig（纯配置，无业务逻辑）
//!
//! 所有字段 Default 值对齐设计文档 08 §三

use oyta_router::impls::routing_strategy::RoutingStrategy;

/// 稀疏训练配置（控制三层稀疏过滤开关）
#[derive(Debug, Clone)]
pub struct SparsityConfig {
    /// 专家级稀疏：每个 batch 只激活 K 个专家（核心算力节省，默认 true）
    pub enable_expert_sparsity: bool,
    /// 路由策略：TopK / Threshold（决定 active_ids）
    pub routing_strategy: RoutingStrategy,
    /// 通道级稀疏：激活专家内部某些通道被 mask（需人工配置 L2 mask，默认 false）
    pub enable_channel_sparsity: bool,
    /// 序列级稀疏：padding 时间步必须 mask（默认 true）
    pub enable_sequence_sparsity: bool,
    /// Debug 开关：强制所有专家激活（关闭 Layer 1 稀疏，默认 false）
    pub force_all_experts_active: bool,
    /// mask 统计日志频率（None = 不输出）
    pub log_mask_every_n: Option<u32>,
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            enable_expert_sparsity: true,
            routing_strategy: RoutingStrategy::TopK { k: 2 },
            enable_channel_sparsity: false,
            enable_sequence_sparsity: true,
            force_all_experts_active: false,
            log_mask_every_n: None,
        }
    }
}

/// 训练主配置（控制 epoch 数 / 梯度累积 / 梯度裁剪）
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// 梯度累积步数（Phase 1 假设 accum=1，简化端到端）
    pub gradient_accumulation_steps: usize,
    /// 最大训练 epoch 数
    pub max_epochs: usize,
    /// 全局梯度裁剪阈值（L2 norm > max_norm → scale grad）
    pub max_grad_norm: f32,
    /// 随机种子（训练可复现）
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self { gradient_accumulation_steps: 1, max_epochs: 10, max_grad_norm: 1.0, seed: 42 }
    }
}
