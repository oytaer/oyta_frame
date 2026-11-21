//! Node：计算图节点 + NodeId 类型别名
//!
//! 稀疏核心：domain_id: Option<u32>
//!   None   = 不属于任何专家（TaskHead / Fusion / Loss / Router）
//!   Some(did) = 属于该领域专家 → sparse_backward 三层过滤用

use std::sync::{Arc, Mutex};
use oyta_tensor::prelude::*;
use crate::types::op::Op;

/// 节点 ID 类型（u64，框架层全局唯一递增，支持大规模训练迭代）
pub type NodeId = u64;

/// 计算图节点（稀疏适配版本）
pub struct Node {
    /// 唯一节点 ID
    pub id: NodeId,
    /// 该节点执行的 Op（Leaf / Linear / Conv1dMasked / ...）
    pub op: Op,
    /// 输入节点 ID 列表（拓扑上游）
    pub inputs: Vec<NodeId>,
    /// 前向输出 Tensor（Arc 共享）
    pub output: Arc<Tensor<f32, Cpu>>,
    /// 是否叶子节点（Parameter / 输入数据）
    pub is_leaf: bool,
    /// 关联的 Parameter（is_leaf=true 时 Some，其他 None）
    pub param: Option<Arc<Mutex<Tensor<f32, Cpu>>>>,

    // ===== v2 稀疏核心 =====
    /// 所属领域：None=不属于任何专家，Some(did)=属于该领域专家
    /// sparse_backward 三层过滤核心字段
    pub domain_id: Option<u32>,
    /// 是否需要梯度（Parameter=true，纯输入数据=false）
    /// backward 时 requires_grad=false 的节点跳过梯度计算
    pub requires_grad: bool,
}

impl Node {
    /// 构造普通算子节点
    pub fn new(
        id: NodeId,
        op: Op,
        inputs: Vec<NodeId>,
        out: Arc<Tensor<f32, Cpu>>,
        did: Option<u32>,
    ) -> Self {
        Self {
            id, op, inputs, output: out,
            is_leaf: false, param: None,
            domain_id: did, requires_grad: true,
        }
    }

    /// 构造叶子节点（Parameter / 输入数据）
    ///
    /// requires_grad=true: Parameter 需要梯度
    /// requires_grad=false: 纯输入数据（如 x_data）不需要梯度
    pub fn leaf(
        id: NodeId,
        out: Tensor<f32, Cpu>,
        param: Option<Arc<Mutex<Tensor<f32, Cpu>>>>,
        did: Option<u32>,
        requires_grad: bool,
    ) -> Self {
        Self {
            id, op: Op::Leaf, inputs: Vec::new(),
            output: Arc::new(out), is_leaf: true,
            param, domain_id: did, requires_grad,
        }
    }
}
