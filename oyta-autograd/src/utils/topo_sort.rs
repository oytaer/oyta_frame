//! topo_reverse：反向拓扑排序（Kahn + BFS）
//!
//! 从 root 节点出发，反向遍历所有可达节点（向上追到叶子 Parameter / 输入）
//! 返回的 Vec<NodeId> 顺序满足：父节点（上游）在子节点（下游）之后
//! （反向遍历需要子节点先算完 backward，才能把梯度传给父节点）

use std::collections::{HashMap, HashSet};
use crate::types::node::{Node, NodeId};

/// 反向拓扑排序：从 root 可达的所有节点按 backward 顺序排列
///
/// Kahn 算法原理：
///   forward[pid] = 所有以 pid 为输入的子节点 → pid 的出边集合
///   反向遍历时"入度为 0"= 没有子节点还没被处理 → 先处理叶子
pub fn topo_reverse(nodes: &HashMap<NodeId, Node>, root_id: NodeId) -> Vec<NodeId> {
    // ===== Step 1: BFS 从 root 向下遍历，构建 forward/reverse 边 =====
    let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new(); // parent → children
    let mut reverse: HashMap<NodeId, Vec<NodeId>> = HashMap::new(); // child → parents
    let mut reachable: HashSet<NodeId> = HashSet::new(); // root 可达的所有节点

    let mut q = vec![root_id];
    reachable.insert(root_id);
    while let Some(cid) = q.pop() {
        if let Some(n) = nodes.get(&cid) {
            for &pid in &n.inputs {
                forward.entry(pid).or_default().push(cid);
                reverse.entry(cid).or_default().push(pid);
                if reachable.insert(pid) { q.push(pid); }
            }
        }
    }

    // ===== Step 2: Kahn 反向拓扑 =====
    // 入度 = forward[parent] 的长度（有多少子节点）
    // 入度为 0 → 没有子节点还没被处理 → 先处理
    let mut deg: HashMap<NodeId, usize> = HashMap::new();
    for &id in &reachable {
        deg.insert(id, forward.get(&id).map(|v| v.len()).unwrap_or(0));
    }

    // 初始队列：所有入度为 0 的节点（backward 最后才处理的）
    let mut kahn_q: Vec<NodeId> = reachable.iter()
        .filter(|&&id| deg.get(&id).copied().unwrap_or(0) == 0)
        .copied().collect();
    kahn_q.sort(); // 确定性顺序

    let mut topo: Vec<NodeId> = Vec::new();
    while let Some(cid) = kahn_q.pop() {
        topo.push(cid);
        // 处理完子节点 → 父节点入度减 1
        if let Some(parents) = reverse.get(&cid) {
            for &pid in parents {
                if let Some(d) = deg.get_mut(&pid) {
                    *d -= 1;
                    if *d == 0 {
                        kahn_q.push(pid);
                    }
                }
            }
        }
        kahn_q.sort_by(|a, b| b.cmp(a)); // 保持确定性
    }

    // Kahn 产物是正向拓扑（先处理下游），reverse 后得到反向（先处理上游父节点）
    topo.reverse();
    topo
}
