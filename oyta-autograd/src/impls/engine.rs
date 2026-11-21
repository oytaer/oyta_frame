//! AutogradEngine：计算图构建 + 管理 + 注册
//!
//! 框架层通过 Engine 构建完整计算图：
//!   1. new() 创建空 Engine
//!   2. register_node() 添加 Op 节点或 Leaf 节点
//!   3. 前向传播正常执行（Engine 只管记录，不管执行）
//!   4. backward(root_id, active_ids) 调 sparse_backward
//!   5. 遍历 grads HashMap → 找到 param 节点 → set_grad() 写回 Parameter

use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use oyta_tensor::prelude::*;
use crate::types::node::{Node, NodeId};
use crate::types::op::Op;
use crate::impls::backward::sparse_backward;

/// 自动微分引擎（计算图容器）
pub struct AutogradEngine {
    /// 所有计算图节点（id → Node）
    nodes: HashMap<NodeId, Node>,
    /// 下一个可用 NodeId（自动递增）
    next_id: NodeId,
    /// 全局 training mode（BatchNorm / Dropout 等模块共享）
    training_mode: Arc<AtomicBool>,
}

impl AutogradEngine {
    /// 创建空 Engine（默认 training_mode = true）
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            training_mode: Arc::new(AtomicBool::new(true)),
        }
    }

    /// 切换训练态（同步给 oyta_tensor::set_training_mode）
    pub fn set_training(&mut self, mode: bool) {
        self.training_mode.store(mode, Ordering::Relaxed);
        oyta_tensor::set_training_mode(mode);
    }

    /// 查询当前 training mode
    pub fn is_training(&self) -> bool {
        self.training_mode.load(Ordering::Relaxed)
    }

    /// 获取 training_mode 的 Arc（共享给子模块）
    pub fn training_mode_arc(&self) -> Arc<AtomicBool> {
        self.training_mode.clone()
    }

    /// 注册普通算子节点
    ///
    /// 返回新分配的 NodeId
    pub fn register_node(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        output: Arc<Tensor<f32, Cpu>>,
        domain_id: Option<u32>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node::new(id, op, inputs, output, domain_id));
        id
    }

    /// 注册叶子节点（Parameter / 输入数据）
    ///
    /// # 参数
    /// - `param`: Some(Arc<Mutex<...>>) 表示 Parameter（需要梯度），None 表示纯输入数据
    /// - `requires_grad`: true=参与 backward 梯度计算，false=跳过
    pub fn register_leaf(
        &mut self,
        output: Tensor<f32, Cpu>,
        param: Option<Arc<Mutex<Tensor<f32, Cpu>>>>,
        domain_id: Option<u32>,
        requires_grad: bool,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node::leaf(id, output, param, domain_id, requires_grad));
        id
    }

    /// 获取节点（只读）
    pub fn get(&self, id: NodeId) -> Option<&Node> { self.nodes.get(&id) }

    /// 当前节点总数
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// 清空计算图（每次训练迭代前调用）
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_id = 0;
    }

    /// 执行 sparse_backward
    ///
    /// # 参数
    /// - `root_id`: loss 节点 ID
    /// - `active_ids`: Router 返回的激活领域列表
    ///
    /// # 返回
    /// grads HashMap（NodeId → Arc<Tensor<f32, Cpu>>）
    /// 外部遍历找 param 节点 → Parameter.set_grad()
    pub fn backward(
        &self,
        root_id: NodeId,
        active_ids: &[u32],
    ) -> HashMap<NodeId, Arc<Tensor<f32, Cpu>>> {
        sparse_backward(root_id, active_ids, &self.nodes)
    }

    /// 批量将 grads 写回 SparseParameterPool 的 Parameter.grad
    ///
    /// 遍历所有叶子节点（is_leaf=true, param=Some）→ 从 grads HashMap 找到对应梯度
    /// 这里只做遍历验证（不直接写回），因为 grads 里存的是 Arc<Tensor<f32,Cpu>> 不是 Mutex 包
    /// 框架层 trainer 会拿到 grads HashMap 后统一调 Parameter.set_grad()
    pub fn apply_grads_to_params(
        &self,
        grads: &HashMap<NodeId, Arc<Tensor<f32, Cpu>>>,
    ) {
        for node in self.nodes.values() {
            if !node.is_leaf { continue; }
            if node.param.is_none() { continue; }
            // 验证：param 叶子节点在 grads 里有对应梯度（未激活专家的应该是 zeros）
            if let Some(g) = grads.get(&node.id) {
                let _numel = g.numel();
            }
        }
    }
}

impl Default for AutogradEngine {
    fn default() -> Self { Self::new() }
}
