//! DomainState：每个领域独立维护的优化器状态（types 层）
//!
//! 为什么按领域独立？
//!   稀疏训练下每个 batch 只激活 K << total 个领域
//!   未激活领域的参数没被 backward → m/v/t 不应该动
//!   如果全局共享 t → 每次 step_sparse 都 +1，但只更新了 K/total 的参数
//!   → 一阶/二阶矩的衰减系数在不同领域上的"有效步数"不一致
//!   → 按领域独立 t 保证每个被激活领域的 Adam 状态正确累积

use std::collections::HashMap;

/// 单个领域的优化器状态（每个领域独立一份，不全局共享）
#[derive(Debug, Default)]
pub struct DomainState {
    /// 一阶矩：key = Parameter.name（与 SparseParameterPool.Parameter.name 完全一致）
    pub m: HashMap<String, Vec<f32>>,
    /// 二阶矩：key = Parameter.name
    pub v: HashMap<String, Vec<f32>>,
    /// 该领域独立的 step 计数器（每次 step_sparse 对该领域执行时 +1）
    pub t: u64,
}

impl DomainState {
    /// 构造空状态（Default）
    pub fn new() -> Self { Self::default() }

    /// 确保指定 Parameter 有足够的 m/v 缓存（首次 step 时调用）
    ///
    /// # 参数
    /// - `name`: Parameter.name（如 "domain_1.mlp.hidden.weight"）
    /// - `numel`: Parameter 的元素数量
    pub fn ensure(&mut self, name: &str, numel: usize) {
        self.m.entry(name.to_string()).or_insert_with(|| vec![0.0f32; numel]);
        self.v.entry(name.to_string()).or_insert_with(|| vec![0.0f32; numel]);
    }

    /// 该领域状态是否已缓存指定 Parameter
    pub fn has_param(&self, name: &str) -> bool {
        self.m.contains_key(name)
    }
}
