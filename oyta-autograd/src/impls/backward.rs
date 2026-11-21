//! sparse_backward：反向拓扑遍历 + domain_id 三层过滤
//!
//! 核心稀疏设计（架构 0.6.8）：
//!   Layer 1: 整专家跳过 — domain_id ∉ active_ids → 给父节点传零梯度 → continue
//!   Layer 2: 通道级 mask — Conv1dMasked / BatchNorm1dMasked 在 op.rs 内部处理
//!   Layer 3: 序列级 mask — Conv1dWithSequenceMask 在 op.rs 内部处理
//!
//! 端到端稀疏数据流：
//!   未激活专家子图被跳过 → Parameter.grad 永远不被写
//!   → Optimizer.step_sparse 遍历 active_ids 时自动跳过它们

use std::collections::HashMap;
use std::sync::Arc;
use oyta_tensor::prelude::*;
use crate::types::node::{Node, NodeId};
use crate::utils::topo_sort::topo_reverse;
use crate::impls::op::{
    Op, compute_backward_op,
    conv1d_masked_backward, batchnorm1d_masked_backward, conv1d_with_seq_mask_backward,
};

/// 稀疏反向传播主入口
///
/// # 参数
/// - `root_id`: loss 节点（计算图终点）
/// - `active_ids`: Router 返回的激活领域列表（稀疏过滤用）
/// - `nodes`: 完整计算图 HashMap
///
/// # 返回
/// grads: HashMap<NodeId, Arc<Tensor<f32, Cpu>>>
///   key = 节点 ID，value = 该节点输出 Tensor 的输入梯度
///   外部（trainer）遍历找到 param 节点 → Parameter.set_grad() 写回
pub fn sparse_backward(
    root_id: NodeId,
    active_ids: &[u32],
    nodes: &HashMap<NodeId, Node>,
) -> HashMap<NodeId, Arc<Tensor<f32, Cpu>>> {
    // ========== Phase 1: 反向拓扑排序 ==========
    let sorted = topo_reverse(nodes, root_id);

    // ========== Phase 2: 初始化 root 梯度 = 全 1 ==========
    // dL/dloss = 1（loss 节点的梯度恒为 1）
    let mut grads: HashMap<NodeId, Arc<Tensor<f32, Cpu>>> = HashMap::new();
    if let Some(loss) = nodes.get(&root_id) {
        grads.insert(root_id, Arc::new(Tensor::<f32, Cpu>::ones(
            loss.output.shape().dims.iter().copied()
        )));
    }

    // ========== Phase 3: 反向遍历每个节点 ==========
    for &cid in &sorted {
        let node = match nodes.get(&cid) { Some(n) => n, None => continue };
        // 取当前节点的 grad_out（已在上一轮 backward 中设置）
        let g_out = match grads.get(&cid) {
            Some(g) => g.clone(),
            None => Arc::new(Tensor::<f32, Cpu>::zeros(
                node.output.shape().dims.iter().copied()
            )),
        };

        // ========== Layer 1: 整专家跳过（domain_id 过滤）==========
        // domain_id=None → 不属于任何专家（TaskHead/Fusion/Loss/Router）→ 正常计算
        // domain_id=Some(did) 且 did ∉ active_ids → 整子图跳过
        let do_skip_entire_subtree = match node.domain_id {
            Some(did) => !active_ids.contains(&did),
            None => false,
        };

        // Layer 1.5: requires_grad=false 的叶子节点跳过（不需要梯度）
        if node.is_leaf && !node.requires_grad {
            // 给父节点传零梯度（该叶子不贡献梯度）
            for &pid in &node.inputs {
                if let Some(pn) = nodes.get(&pid) {
                    grads.insert(pid, Arc::new(Tensor::<f32, Cpu>::zeros(
                        pn.output.shape().dims.iter().copied()
                    )));
                }
            }
            continue;
        }

        if do_skip_entire_subtree {
            // 给所有父节点传零梯度 → 自然向上传播 → 整子图 backward 终止
            for &pid in &node.inputs {
                if let Some(pn) = nodes.get(&pid) {
                    grads.insert(pid, Arc::new(Tensor::<f32, Cpu>::zeros(
                        pn.output.shape().dims.iter().copied()
                    )));
                }
            }
            continue; // ⚠️ 关键：跳过这个节点自己的 backward 计算
        }

        // ========== 正常 backward（domain_id=None 或 ∈ active_ids）==========
        let inputs: Vec<Arc<Tensor<f32, Cpu>>> = node.inputs.iter()
            .filter_map(|pid| nodes.get(pid).map(|p| p.output.clone()))
            .collect();

        // 根据 Op 类型分发到对应的 backward 实现
        let input_grads = match &node.op {
            // Layer 2 mask 分支
            Op::Conv1dMasked { channel_mask } => {
                conv1d_masked_backward(&g_out, &inputs, channel_mask)
            }
            Op::BatchNorm1dMasked { channel_mask } => {
                batchnorm1d_masked_backward(&g_out, &inputs, channel_mask)
            }
            // Layer 3 mask 分支
            Op::Conv1dWithSequenceMask { sequence_mask } => {
                conv1d_with_seq_mask_backward(&g_out, &inputs, sequence_mask)
            }
            // 不可训练节点
            Op::Router { .. } | Op::Leaf => Vec::new(),
            // 标准算子
            other => compute_backward_op(other, &g_out, &inputs),
        };

        // ========== Phase 4: 累加到父节点梯度 ==========
        // 一个 Parameter 被多个算子使用 → 多次累加
        for (i, &pid) in node.inputs.iter().enumerate() {
            if i >= input_grads.len() { break; }
            let new_g = input_grads[i].clone();
            match grads.get_mut(&pid) {
                Some(existing) => {
                    // Vec 逐元素加（两个子节点的梯度累加）
                    let ed = existing.data_slice(); let nd = new_g.data_slice();
                    let sum: Vec<f32> = ed.iter().zip(nd.iter()).map(|(a, b)| a + b).collect();
                    let shape = existing.shape().dims.clone();
                    *existing = Arc::new(Tensor::<f32, Cpu>::from_data(sum, shape).unwrap());
                }
                None => { grads.insert(pid, new_g); }
            }
        }
    }

    grads
}
