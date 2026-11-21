//! MultiDomainLoss：主损失 + 只给激活领域加辅助损失
//!
//! 架构 0.6.8 三层防线：
//!   Layer 1: Router → 返回 active_ids（哪些领域激活）
//!   Layer 2: MultiDomainLoss.compute → 只对 active_ids 内的领域算辅助损失
//!   Layer 3: sparse_backward → domain_id ∉ active_ids 整子图跳过
//!
//! 数学：
//!   L = main(fused, target) + aux_weight × Σ_{did ∈ active ∩ aux_losses} aux[did](expert_out[did], target)
//!
//! 关键约束：
//!   ❌ 绝对不能遍历所有 aux_losses.keys() 然后不管 active 都算
//!      → 会对未激活领域算 loss，但它们的 Parameter 接收不到梯度
//!   ✅ 必须遍历 expert_outs（已过滤：只含激活领域输出）
//!      → 二次 active.contains(did) 确认（防御性检查）

use std::collections::HashMap;
use oyta_tensor::prelude::*;
use crate::traits::loss_fn::LossFn;

/// 多领域联合损失（主损失 + 按 active_ids 过滤的辅助损失）
pub struct MultiDomainLoss {
    /// 主损失：所有激活专家融合后的 fused → target
    pub main_loss: Box<dyn LossFn>,
    /// 辅助损失表：domain_id → LossFn（可选，不是每个领域都有 aux）
    pub aux_losses: HashMap<u32, Box<dyn LossFn>>,
    /// 辅助损失权重（默认 0.1）
    pub aux_weight: f32,
}

impl MultiDomainLoss {
    /// 构造 MultiDomainLoss（只指定主损失，aux_weight 默认 0.1）
    pub fn new(main: Box<dyn LossFn>) -> Self {
        Self { main_loss: main, aux_losses: HashMap::new(), aux_weight: 0.1 }
    }

    /// 注册某个领域的辅助损失
    pub fn with_auxiliary(&mut self, did: u32, loss: Box<dyn LossFn>) {
        self.aux_losses.insert(did, loss);
    }

    /// 设置辅助损失权重
    pub fn with_aux_weight(mut self, w: f32) -> Self {
        self.aux_weight = w;
        self
    }

    /// 计算总损失（主损失 + 只对 active_ids 内领域的辅助损失）
    ///
    /// # 参数
    /// - `fused`: 融合后的总输出（所有激活专家融合）
    /// - `target`: 标签
    /// - `active`: Router 返回的激活领域列表
    /// - `expert_outs`: 只含激活领域的专家输出（框架层已过滤）
    pub fn compute(
        &self,
        fused: &Tensor<f32, Cpu>,
        target: &Tensor<f32, Cpu>,
        active: &[u32],
        expert_outs: &[(u32, Tensor<f32, Cpu>)],
    ) -> Tensor<f32, Cpu> {
        // ===== Step 1: 主损失（始终对 fused 计算，不稀疏化）=====
        let main_val = self.main_loss.item(&self.main_loss.compute(fused, target));
        let mut total = main_val;

        // ===== Step 2: 辅助损失（只对 active_ids 内的领域）=====
        if self.aux_weight > 0.0 {
            // 遍历 expert_outs（框架层已过滤：只含激活领域输出）
            for (did, out) in expert_outs {
                // 二次确认：did ∈ active（防御性检查）
                if !active.contains(did) { continue; }
                if let Some(aux) = self.aux_losses.get(did) {
                    total += self.aux_weight * aux.item(&aux.compute(out, target));
                }
                // ⚠️ 未激活领域的 aux_loss 不参与 total → 它们的 Parameter 接收不到梯度
                // 这和 autograd sparse_backward + optim step_sparse 的过滤是一致的三层防线
            }
        }

        Tensor::<f32, Cpu>::from_data(vec![total], [1]).unwrap()
    }

    /// 把 loss Tensor 转 f32 标量（底层 Tensor.mean(None) → D::reduce_mean）
    pub fn item(&self, loss: &Tensor<f32, Cpu>) -> f32 {
        if loss.numel() == 1 { return loss.item(); }
        loss.mean(None).expect("mean 底层调用失败").item()
    }
}
