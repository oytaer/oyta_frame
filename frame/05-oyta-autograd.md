# oyta-autograd 开发文档（Phase 5）

> crate：`oyta-autograd/` | 依赖：**`oyta-tensor = "0.1.1"`（唯一底层！）** | 职责：Node + Op（含 4 个掩码变体）+ sparse_backward（domain_id 过滤）

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-autograd"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
rayon       = "1.12.0"
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs

```rust
//! oyta-autograd：自动微分引擎（稀疏适配）
//!
//! 核心稀疏设计：
//!   Node 带 domain_id: Option<u32>
//!   sparse_backward 反向遍历时：node.domain_id ∉ active_ids → 跳过整 Op backward
//!
//! 底层算子 backward 全调 oyta-tensor Backend trait（不手写 matmul / layernorm）

pub mod node;
pub mod op;
pub mod backward;
pub mod engine;

pub use node::{Node, NodeId};
pub use op::{Op, compute_backward_op};
pub use backward::sparse_backward;
pub use engine::AutogradEngine;

pub mod prelude {
    pub use crate::{Node, NodeId, Op, compute_backward_op, sparse_backward, AutogradEngine};
    pub use oyta_tensor::prelude::*;
    pub use std::collections::HashMap;
    pub use std::sync::{Arc, Mutex};
}
```

---

# 三、src/node.rs

```rust
//! Node：计算图节点
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use oyta_tensor::prelude::*;
use super::op::Op;

pub type NodeId = u32;

pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub output: Arc<Tensor<f32, Cpu>>,
    pub is_leaf: bool,
    pub param: Option<Arc<Mutex<Tensor<f32, Cpu>>>>,

    // ===== v2 稀疏核心 =====
    /// 所属领域 None = 不属于任何专家（TaskHead / Fusion / Loss）
    ///                     Some(did) = 属于该领域专家
    /// sparse_backward 过滤用
    pub domain_id: Option<u32>,
}

impl Node {
    pub fn new(id: NodeId, op: Op, inputs: Vec<NodeId>, out: Arc<Tensor<f32, Cpu>>, did: Option<u32>) -> Self {
        Self { id, op, inputs, output: out, is_leaf: false, param: None, domain_id: did }
    }
    pub fn leaf(id: NodeId, out: Tensor<f32, Cpu>, param: Arc<Mutex<Tensor<f32, Cpu>>>, did: Option<u32>) -> Self {
        Self { id, op: Op::Leaf, inputs: Vec::new(), output: Arc::new(out),
               is_leaf: true, param: Some(param), domain_id: did }
    }
}
```

---

# 四、src/op.rs（Op 枚举 + compute_backward，底层算子全调 oyta-tensor）

```rust
//! Op 枚举 + compute_backward 统一入口
//!
//! 算子 backward 底层调用 oyta-tensor Backend trait（不手写 matmul / layernorm / flash_attn）
//! Phase 2 新增 4 个带掩码变体：
//!   Conv1dMasked         → 内部先 mask grad_out 通道 → 调 Backend Conv1d backward
//!   BatchNorm1dMasked    → 先调 Backend BN backward → 后 mask 清零被 mask 特征
//!   Conv1dWithSequenceMask → 先调 Backend Conv1d backward → 后 mask padding 时间步
//!   Router               → Vec::new()（不可训练）

use std::collections::HashMap;
use std::sync::Arc;
use oyta_tensor::prelude::*;
use super::node::NodeId;

#[derive(Debug, Clone)]
pub enum Op {
    Leaf,

    // ===== 基础算子（backward 全调 oyta-tensor）=====
    Linear,
    Matmul { transa: bool, transb: bool },
    Relu, Gelu, Sigmoid, Tanh, Silu,
    Softmax { dim: isize },
    LayerNorm { eps: f32 },
    RMSNorm { eps: f32 },

    // ===== Phase 2：带掩码的稀疏变体 =====
    Conv1dMasked { channel_mask: Vec<f32> },
    BatchNorm1dMasked { channel_mask: Vec<f32> },
    Conv1dWithSequenceMask { sequence_mask: Vec<f32> },
    Router { domain_mask: Vec<bool>, normalized_weights: Vec<f32> },

    // ===== Loss =====
    CrossEntropy { num_classes: usize },
    MSE,
}

/// compute_backward：统一入口（底层算子调 oyta-tensor Backend）
pub fn compute_backward_op(
    op: &Op,
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    // Router：不可训练
    if matches!(op, Op::Router { .. }) { return Vec::new(); }
    if matches!(op, Op::Leaf) { return Vec::new(); }

    // ===== Linear backward =====
    if matches!(op, Op::Linear) && input_tensors.len() >= 3 {
        let x = &input_tensors[0]; let w = &input_tensors[1]; let b = &input_tensors[2];
        // dL/dx = grad_out @ W
        let dldx = grad_out.matmul(w).unwrap();  // 直接调 oyta_tensor matmul
        // dL/dw = grad_out^T @ x
        let gwt = Self::transpose(&grad_out);
        let dldw = gwt.matmul(x).unwrap();
        // dL/db = sum over batch（reduce_sum，oyta_tensor 应该有）
        let dldb = b.clone(); // 简化：placeholder
        return vec![Arc::new(dldx), Arc::new(dldw), Arc::new(dldb)];
    }

    // ===== Relu backward =====
    if matches!(op, Op::Relu) && !input_tensors.is_empty() {
        let x = input_tensors[0].data_slice(); let go = grad_out.data_slice();
        let mut dx = vec![0.0f32; x.len()];
        for i in 0..x.len() { if x[i] > 0.0 { dx[i] = go[i]; } }
        return vec![Arc::new(Tensor::<f32, Cpu>::from_data(dx, input_tensors[0].shape().clone()).unwrap())];
    }

    // ===== 其他 Op：Phase 1 骨架，底层算子都通过 oyta-tensor 调用 =====
    // 占位返回零梯度（实际实现 Phase 1.5 补）
    input_tensors.iter().map(|t| {
        Arc::new(Tensor::<f32, Cpu>::zeros(t.shape().dims.iter().copied()))
    }).collect()
}

// ===== Conv1dMasked backward（L2 稀疏）=====
pub fn conv1d_masked_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    channel_mask: &[f32],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    let x = &input_tensors[0]; let w = &input_tensors[1];
    let go_d = grad_out.data_slice();
    let batch = grad_out.shape().dims[0];
    let out_c = grad_out.shape().dims[1];
    let out_len = grad_out.shape().dims[2];
    let in_c = x.shape().dims[1];
    let seq_len = x.shape().dims[2];
    let k_size = w.shape().dims[2];

    // Step 1：先把 grad_out 被 mask 通道清零（inv_mask = 1 - mask）
    let inv_mask: Vec<f32> = channel_mask.iter().map(|&m| 1.0 - m).collect();
    let mut masked_go = vec![0.0f32; grad_out.numel()];
    for n in 0..batch { for oc in 0..out_c {
        let scale = inv_mask[oc];
        if scale == 0.0 { continue; }
        let row = n * out_c * out_len + oc * out_len;
        for t in 0..out_len { masked_go[row + t] = go_d[row + t] * scale; }
    }}

    // Step 2：标准 Conv1d backward（互相关 ∂w + dilated 反卷积 ∂x）
    // 这是算子本身的 backward——如果 oyta-device Backend trait 提供了 conv1d_backward 就直接调
    // Phase 1 骨架：手动实现（纯 Vec 操作，验证正确性）
    let x_d = x.data_slice(); let w_d = w.data_slice();
    let mut gx = vec![0.0f32; batch * in_c * seq_len];
    let mut gw = vec![0.0f32; out_c * in_c * k_size];
    let mut gb = vec![0.0f32; out_c];

    // ∂L/∂b[oc]
    for oc in 0..out_c {
        let mut acc = 0.0f64;
        for n in 0..batch { for t in 0..out_len {
            acc += masked_go[n*out_c*out_len + oc*out_len + t] as f64;
        }}
        gb[oc] = acc as f32;
    }
    // ∂L/∂w[oc, ic, k]（互相关）
    for oc in 0..out_c { for ic in 0..in_c { for k in 0..k_size {
        let mut acc = 0.0f64;
        for n in 0..batch { for t in 0..out_len {
            let in_idx = t + k - 1;  // stride=1, padding=1
            if in_idx < seq_len {
                acc += (masked_go[n*out_c*out_len+oc*out_len+t] as f64)
                     * (x_d[n*in_c*seq_len+ic*seq_len+in_idx] as f64);
            }
        }}
        gw[oc*in_c*k_size + ic*k_size + k] = acc as f32;
    }}}
    // ∂L/∂x[n, ic, t]（dilated 反卷积）
    for n in 0..batch { for ic in 0..in_c { for t in 0..seq_len {
        let mut acc = 0.0f64;
        for oc in 0..out_c { for t_out in 0..out_len {
            let k = t + 1 - t_out;  // stride=1, padding=1
            if k >= 0 && k < k_size {
                acc += (masked_go[n*out_c*out_len+oc*out_len+t_out] as f64)
                     * (w_d[oc*in_c*k_size+ic*k_size+k] as f64);
            }
        }}
        gx[n*in_c*seq_len + ic*seq_len + t] = acc as f32;
    }}}

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, in_c, seq_len]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gw, [out_c, in_c, k_size]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [out_c]).unwrap()),
    ]
}

// ===== BatchNorm1dMasked backward（L2 稀疏 + 完整简化公式）=====
pub fn batchnorm1d_masked_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    channel_mask: &[f32],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    // input_tensors: [x, cached_xhat, cached_inv_std, gamma]
    let go_d = grad_out.data_slice();
    let xhat_d = input_tensors[1].data_slice();
    let inv_std_d = input_tensors[2].data_slice();
    let gamma_d = input_tensors[3].data_slice();

    let batch = grad_out.shape().dims[0];
    let nf = grad_out.shape().dims[1];
    let inv_n: f64 = 1.0 / batch as f64;

    let mut gx = vec![0.0f32; batch * nf];
    let mut gg = vec![0.0f32; nf];
    let mut gb = vec![0.0f32; nf];

    // 完整简化公式：dL/dx[m,f] = inv_std[f] × [ scaled_go[m] - mean_sgo - xhat[m] × mean_sx ]
    for f in 0..nf {
        let iv = inv_std_d[f] as f64;
        let g_f64 = gamma_d[f] as f64;
        let mut grad_beta: f64 = 0.0;
        let mut grad_gamma: f64 = 0.0;
        let mut sx_sum: f64 = 0.0;
        let mut sgo: Vec<f64> = Vec::with_capacity(batch);
        for n in 0..batch {
            let idx = n * nf + f;
            let go = go_d[idx] as f64;
            let xh = xhat_d[idx] as f64;
            grad_beta += go;
            grad_gamma += go * xh;
            let sg = go * g_f64;
            sgo.push(sg);
            sx_sum += sg * xh;
        }
        // L2 mask：被 mask 特征清零（梯度完全不传）
        if channel_mask[f] >= 0.5 {
            // grad_beta / grad_gamma 对这个特征不应该贡献梯度
            // 但 Beta / Gamma 是共享参数，这里置零
            gb[f] = 0.0;
            gg[f] = 0.0;
            for n in 0..batch { gx[n * nf + f] = 0.0; }
            continue;
        }
        gb[f] = grad_beta as f32;
        gg[f] = grad_gamma as f32;
        let mean_sgo: f64 = sgo.iter().sum::<f64>() * inv_n;
        let mean_sx: f64 = sx_sum * inv_n;
        for n in 0..batch {
            let idx = n * nf + f;
            let xh = xhat_d[idx] as f64;
            let bracket: f64 = sgo[n] - mean_sgo - xh * mean_sx;
            gx[idx] = (iv * bracket) as f32;
        }
    }

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, nf]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gg, [nf]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [nf]).unwrap()),
    ]
}

// ===== Conv1dWithSequenceMask backward（L3 稀疏）=====
pub fn conv1d_with_seq_mask_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    sequence_mask: &[f32],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    // 先算标准 Conv1d backward（同 conv1d_masked_backward 里的 ∂x 部分）
    // 然后 padding 时间步清零 gx[:, :, t]
    let x = &input_tensors[0]; let w = &input_tensors[1];
    let go_d = grad_out.data_slice(); let x_d = x.data_slice(); let w_d = w.data_slice();
    let batch = grad_out.shape().dims[0]; let out_c = grad_out.shape().dims[1]; let out_len = grad_out.shape().dims[2];
    let in_c = x.shape().dims[1]; let seq_len = x.shape().dims[2]; let k_size = w.shape().dims[2];

    let mut gx = vec![0.0f32; batch * in_c * seq_len];
    let mut gw = vec![0.0f32; out_c * in_c * k_size];
    let mut gb = vec![0.0f32; out_c];
    for oc in 0..out_c {
        let mut acc = 0.0f64;
        for n in 0..batch { for t in 0..out_len { acc += go_d[n*out_c*out_len+oc*out_len+t] as f64; }}
        gb[oc] = acc as f32;
    }
    for oc in 0..out_c { for ic in 0..in_c { for k in 0..k_size {
        let mut acc = 0.0f64;
        for n in 0..batch { for t in 0..out_len {
            let ii = t + k - 1;
            if ii < seq_len { acc += (go_d[n*out_c*out_len+oc*out_len+t] as f64) * (x_d[n*in_c*seq_len+ic*seq_len+ii] as f64); }
        }}
        gw[oc*in_c*k_size+ic*k_size+k] = acc as f32;
    }}}
    for n in 0..batch { for ic in 0..in_c { for t in 0..seq_len {
        let mut acc = 0.0f64;
        for oc in 0..out_c { for t_out in 0..out_len {
            let k = t + 1 - t_out;
            if k >= 0 && k < k_size { acc += (go_d[n*out_c*out_len+oc*out_len+t_out] as f64) * (w_d[oc*in_c*k_size+ic*k_size+k] as f64); }
        }}
        gx[n*in_c*seq_len+ic*seq_len+t] = acc as f32;
    }}}

    // L3 mask：清零 padding 时间步的 grad_x
    for t in 0..seq_len {
        if sequence_mask[t] >= 0.5 {
            for n in 0..batch { for ic in 0..in_c { gx[n*in_c*seq_len+ic*seq_len+t] = 0.0; }}
        }
    }

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, in_c, seq_len]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gw, [out_c, in_c, k_size]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [out_c]).unwrap()),
    ]
}
```

---

# 五、src/backward.rs（sparse_backward 核心 domain_id 过滤）

```rust
//! sparse_backward：反向拓扑遍历 + domain_id 过滤
//!
//! 核心稀疏判断：
//!   if let Some(did) = node.domain_id {
//!     if !active_ids.contains(&did) {
//!       // 给所有父节点传零梯度 → 整子图断开
//!       continue;
//!     }
//!   }

use std::collections::HashMap;
use std::sync::Arc;
use oyta_tensor::prelude::*;
use super::node::{Node, NodeId};
use super::op::{Op, conv1d_masked_backward, batchnorm1d_masked_backward, conv1d_with_seq_mask_backward};

pub fn sparse_backward(
    root_id: NodeId,
    active_ids: &[u32],
    nodes: &HashMap<NodeId, Node>,
) -> HashMap<NodeId, Arc<Tensor<f32, Cpu>>> {
    // Step 1：反向拓扑排序（Kahn + BFS 从 root 向下）
    let sorted = topo_reverse(nodes, root_id);

    // Step 2：初始化 root grad = 全 1
    let mut grads: HashMap<NodeId, Arc<Tensor<f32, Cpu>>> = HashMap::new();
    if let Some(loss) = nodes.get(&root_id) {
        grads.insert(root_id, Arc::new(Tensor::<f32, Cpu>::ones(loss.output.shape().dims.iter().copied())));
    }

    // Step 3：反向遍历
    for &cid in &sorted {
        let node = match nodes.get(&cid) { Some(n) => n, None => continue };
        let g_out = match grads.get(&cid) { Some(g) => g.clone(), None => {
            Arc::new(Tensor::<f32, Cpu>::zeros(node.output.shape().dims.iter().copied()))
        }};

        // ===== 核心稀疏判断 =====
        let do_skip = if let Some(did) = node.domain_id {
            !active_ids.contains(&did)
        } else { false };

        if do_skip {
            // 给所有父节点传零梯度 → 整子图断开
            for &pid in &node.inputs {
                if let Some(pn) = nodes.get(&pid) {
                    grads.insert(pid, Arc::new(Tensor::<f32, Cpu>::zeros(pn.output.shape().dims.iter().copied())));
                }
            }
            continue;
        }

        // ===== Op.backward 分发 =====
        let inputs: Vec<Arc<Tensor<f32, Cpu>>> = node.inputs.iter()
            .filter_map(|pid| nodes.get(pid).map(|p| p.output.clone())).collect();

        let input_grads = match &node.op {
            Op::Conv1dMasked { channel_mask } => conv1d_masked_backward(&g_out, &inputs, channel_mask),
            Op::BatchNorm1dMasked { channel_mask } => batchnorm1d_masked_backward(&g_out, &inputs, channel_mask),
            Op::Conv1dWithSequenceMask { sequence_mask } => conv1d_with_seq_mask_backward(&g_out, &inputs, sequence_mask),
            Op::Router { .. } => Vec::new(),
            other => compute_backward_op(other, &g_out, &inputs),
        };

        // ===== 累加父节点梯度 =====
        for (i, &pid) in node.inputs.iter().enumerate() {
            if i >= input_grads.len() { break; }
            let new_g = input_grads[i].clone();
            match grads.get_mut(&pid) {
                Some(existing) => {
                    let ed = existing.data_slice(); let nd = new_g.data_slice();
                    let sum: Vec<f32> = ed.iter().zip(nd.iter()).map(|(a,b)| a+b).collect();
                    let shape = existing.shape().dims.clone();
                    *existing = Arc::new(Tensor::<f32, Cpu>::from_data(sum, shape).unwrap());
                }
                None => { grads.insert(pid, new_g); }
            }
        }
    }
    grads
}

// ===== 反向拓扑排序（BFS + Kahn）=====
fn topo_reverse(nodes: &HashMap<NodeId, Node>, root_id: NodeId) -> Vec<NodeId> {
    let mut reverse: HashMap<NodeId, Vec<NodeId>> = HashMap::new();   // child → parents
    let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();   // parent → children
    let mut reachable: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
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
    // Kahn：入度为 0 的 = forward[parent] 长度（即有多少子节点）
    let mut deg: HashMap<NodeId, usize> = HashMap::new();
    for &id in &reachable {
        deg.insert(id, forward.get(&id).map(|v| v.len()).unwrap_or(0));
    }
    let mut kahn_q: Vec<NodeId> = reachable.iter()
        .filter(|&&id| deg.get(&id).copied().unwrap_or(0) == 0)
        .copied().collect();
    kahn_q.sort();
    let mut topo: Vec<NodeId> = Vec::new();
    while let Some(cid) = kahn_q.pop() {
        topo.push(cid);
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
        kahn_q.sort_by(|a, b| b.cmp(a));
    }
    topo.reverse();
    topo
}
```

---

# 六、oyta-autograd 验收标准

```
[ ] cargo check --package oyta-autograd 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1"（无 rust-version、无 path）
[ ] Op 枚举有 4 个掩码变体（Conv1dMasked / BatchNorm1dMasked / Conv1dWithSequenceMask / Router）
[ ] sparse_backward 核心 domain_id 过滤 continue 整子图跳过
[ ] Conv1dMasked backward 手动实现完整 ∂b / ∂w（互相关）/ ∂x（dilated 反卷积）+ L2 mask
[ ] BatchNorm1dMasked backward 完整简化公式 + L2 mask 清零被 mask 特征
[ ] Router Op 返回 Vec::new()
[ ] 每个函数有中文注释
```

---

# 七、Op 掩码变体 backward 分支（架构 0.6.8 完整逻辑）

> 本节补齐 op.rs 中三个掩码变体的完整 backward 实现逻辑（架构设计文档 §0.6.8 定义）。当前文档 §四 已有骨架，本节细化每个分支的精确 mask 应用顺序和梯度传递规则。

## 7.1 Conv1dMasked（Level 2 通道稀疏）

```rust
//! Conv1dMasked backward 完整逻辑（必须严格按此顺序执行）
//!
//! 稀疏语义：channel_mask[f] = 1.0 → 第 f 个输出通道被 mask → 该通道梯度清零
//! 稀疏位置：backward 前对 grad_out 做通道级缩放（pre-mask），然后调标准 Conv1d backward
//!
//! 为什么在 grad_out 上 pre-mask？
//!   ∂L/∂w[oc, ic, k] = Σ_{n, o} grad_out_masked[n, oc, o] × x[n, ic, ...]
//!   如果 grad_out_masked[oc, :] = 0 → 该 out_channel 对 w 的梯度自动为零
//!   ∂L/∂x[n, ic, t] = Σ_{oc} grad_out_masked[n, oc, ...] × w[oc, ic, k]
//!   被 mask 的 out_channel 不贡献梯度 → 传播回输入的梯度也是正确稀疏的

pub fn conv1d_masked_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    channel_mask: &[f32],  // mask[f] = 0.0→激活 1.0→mask（与 Router 定义一致）
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    let x = &input_tensors[0]; let w = &input_tensors[1];
    let batch = grad_out.shape().dims[0];
    let out_c = grad_out.shape().dims[1];
    let out_len = grad_out.shape().dims[2];
    let in_c = x.shape().dims[1];
    let seq_len = x.shape().dims[2];
    let k_size = w.shape().dims[2];

    // ========== Step 1: grad_out 通道级 pre-mask ==========
    // inv_mask[f] = 1 - channel_mask[f]：被 mask 的通道 inv_mask = 0
    let go_d = grad_out.data_slice();
    let inv_mask: Vec<f32> = channel_mask.iter().map(|&m| 1.0 - m).collect();
    let mut masked_go = vec![0.0f32; grad_out.numel()];
    for n in 0..batch {
        for oc in 0..out_c {
            let scale = inv_mask[oc];
            if scale == 0.0 { continue; }  // 整通道跳过
            let row = n * out_c * out_len + oc * out_len;
            for t in 0..out_len {
                masked_go[row + t] = go_d[row + t] * scale;
            }
        }
    }
    // ⚠️ masked_go 替代原始 grad_out 进入后续所有计算

    // ========== Step 2: 标准 Conv1d backward（用 masked_go 替代 grad_out）==========
    let x_d = x.data_slice(); let w_d = w.data_slice();
    let mut gx = vec![0.0f32; batch * in_c * seq_len];
    let mut gw = vec![0.0f32; out_c * in_c * k_size];
    let mut gb = vec![0.0f32; out_c];

    // ∂L/∂b[oc] = Σ masked_go[n, oc, o]（只有未 mask 通道有值）
    for oc in 0..out_c {
        let mut acc: f64 = 0.0;
        for n in 0..batch {
            for t in 0..out_len {
                acc += masked_go[n * out_c * out_len + oc * out_len + t] as f64;
            }
        }
        gb[oc] = acc as f32;
    }

    // ∂L/∂w[oc, ic, k]（互相关）
    for oc in 0..out_c {
        for ic in 0..in_c {
            for k in 0..k_size {
                let mut acc: f64 = 0.0;
                for n in 0..batch {
                    for t in 0..out_len {
                        let in_idx = t + k - 1;  // stride=1, padding=1
                        if in_idx < seq_len {
                            acc += (masked_go[n * out_c * out_len + oc * out_len + t] as f64)
                                 * (x_d[n * in_c * seq_len + ic * seq_len + in_idx] as f64);
                        }
                    }
                }
                gw[oc * in_c * k_size + ic * k_size + k] = acc as f32;
            }
        }
    }

    // ∂L/∂x[n, ic, t]（dilated 反卷积）
    for n in 0..batch {
        for ic in 0..in_c {
            for t in 0..seq_len {
                let mut acc: f64 = 0.0;
                for oc in 0..out_c {
                    for t_out in 0..out_len {
                        let k = t + 1 - t_out;  // stride=1, padding=1
                        if k >= 0 && k < k_size {
                            acc += (masked_go[n * out_c * out_len + oc * out_len + t_out] as f64)
                                 * (w_d[oc * in_c * k_size + ic * k_size + k] as f64);
                        }
                    }
                }
                gx[n * in_c * seq_len + ic * seq_len + t] = acc as f32;
            }
        }
    }

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, in_c, seq_len]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gw, [out_c, in_c, k_size]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [out_c]).unwrap()),
    ]
}
```

## 7.2 BatchNorm1dMasked（Level 2 通道稀疏）

```rust
//! BatchNorm1dMasked backward 完整逻辑（架构 0.6.8）
//!
//! 稀疏语义：被 mask 的特征 channel f → grad_x[:, f] 全部清零 + grad_gamma[f] + grad_beta[f] 清零
//! 稀疏位置：标准 BN backward 计算完后，对被 mask 的特征做 post-zero
//!
//! 为什么 post-zero 而不是 pre-mask？
//!   BN backward 内部有 mean_sgo / mean_sx 全局聚合
//!   如果 pre-mask grad_out → 聚合均值会变小（分母不变），反而不正确
//!   正确做法：先算完整 BN backward（所有特征），然后对 mask 特征整体置零

pub fn batchnorm1d_masked_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    channel_mask: &[f32],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    // input_tensors 约定：[x_placeholder, cached_xhat, cached_inv_std, gamma]
    let xhat_d = input_tensors[1].data_slice();
    let inv_std_d = input_tensors[2].data_slice();
    let gamma_d = input_tensors[3].data_slice();
    let go_d = grad_out.data_slice();

    let batch = grad_out.shape().dims[0];
    let nf = grad_out.shape().dims[1];
    let inv_n: f64 = 1.0 / batch as f64;

    let mut gx = vec![0.0f32; batch * nf];
    let mut gg = vec![0.0f32; nf];
    let mut gb = vec![0.0f32; nf];

    // ========== 标准 BN backward 简化公式 ==========
    for f in 0..nf {
        let iv = inv_std_d[f] as f64;
        let g_f64 = gamma_d[f] as f64;
        let mut grad_beta: f64 = 0.0;
        let mut grad_gamma: f64 = 0.0;
        let mut sx_sum: f64 = 0.0;
        let mut sgo: Vec<f64> = Vec::with_capacity(batch);

        for n in 0..batch {
            let idx = n * nf + f;
            let go = go_d[idx] as f64;
            let xh = xhat_d[idx] as f64;
            grad_beta += go;
            grad_gamma += go * xh;
            let sg = go * g_f64;
            sgo.push(sg);
            sx_sum += sg * xh;
        }

        // ========== L2 mask 判断：被 mask 特征直接置零（跳过完整 BN 公式）==========
        if channel_mask[f] >= 0.5 {
            gb[f] = 0.0;
            gg[f] = 0.0;
            for n in 0..batch { gx[n * nf + f] = 0.0; }
            continue;  // ⚠️ 关键：被 mask 特征不参与 mean_sgo / mean_sx 聚合
        }

        // 未 mask 特征：完整 BN 简化公式
        gb[f] = grad_beta as f32;
        gg[f] = grad_gamma as f32;
        let mean_sgo: f64 = sgo.iter().sum::<f64>() * inv_n;
        let mean_sx: f64 = sx_sum * inv_n;
        for n in 0..batch {
            let idx = n * nf + f;
            let xh = xhat_d[idx] as f64;
            let bracket: f64 = sgo[n] - mean_sgo - xh * mean_sx;
            gx[idx] = (iv * bracket) as f32;
        }
    }

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, nf]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gg, [nf]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [nf]).unwrap()),
    ]
}
```

## 7.3 Conv1dWithSequenceMask（Level 3 序列稀疏）

```rust
//! Conv1dWithSequenceMask backward 完整逻辑（架构 0.6.8）
//!
//! 稀疏语义：sequence_mask[t] = 1.0 → 时间步 t 是 padding → 该时间步的 grad_x 清零
//! 稀疏位置：标准 Conv1d backward 计算完后 post-zero padding 时间步
//!
//! 为什么 post-zero 而不是 pre-mask？
//!   卷积内部感受野交叉，padding 位置可能影响有效位置的 grad_out
//!   pre-mask grad_out 会让 padding 不参与 grad_w（这其实合理）
//!   但 post-zero grad_x 才保证 padding 时间步梯度不向上游传播
//!   本实现采用：先算完整 Conv1d backward → 再 post-zero grad_x 的 padding 时间步

pub fn conv1d_with_seq_mask_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    sequence_mask: &[f32],  // mask[t] = 0.0→有效 1.0→padding
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    let x = &input_tensors[0]; let w = &input_tensors[1];
    let go_d = grad_out.data_slice(); let x_d = x.data_slice(); let w_d = w.data_slice();
    let batch = grad_out.shape().dims[0]; let out_c = grad_out.shape().dims[1];
    let out_len = grad_out.shape().dims[2];
    let in_c = x.shape().dims[1]; let seq_len = x.shape().dims[2];
    let k_size = w.shape().dims[2];

    // ========== 完整 Conv1d backward（同 conv1d_masked_backward 的 Step 2）==========
    let mut gx = vec![0.0f32; batch * in_c * seq_len];
    let mut gw = vec![0.0f32; out_c * in_c * k_size];
    let mut gb = vec![0.0f32; out_c];

    // ∂b, ∂w, ∂x（完整公式，略同 Conv1dMasked Step 2）
    for oc in 0..out_c { let mut acc: f64 = 0.0;
        for n in 0..batch { for t in 0..out_len { acc += go_d[n*out_c*out_len+oc*out_len+t] as f64; }}
        gb[oc] = acc as f32; }
    for oc in 0..out_c { for ic in 0..in_c { for k in 0..k_size { let mut acc: f64 = 0.0;
        for n in 0..batch { for t in 0..out_len {
            let ii = t + k - 1;
            if ii < seq_len { acc += (go_d[n*out_c*out_len+oc*out_len+t] as f64) * (x_d[n*in_c*seq_len+ic*seq_len+ii] as f64); }
        }}
        gw[oc*in_c*k_size+ic*k_size+k] = acc as f32;
    }}}
    for n in 0..batch { for ic in 0..in_c { for t in 0..seq_len { let mut acc: f64 = 0.0;
        for oc in 0..out_c { for t_out in 0..out_len {
            let k = t + 1 - t_out;
            if k >= 0 && k < k_size { acc += (go_d[n*out_c*out_len+oc*out_len+t_out] as f64) * (w_d[oc*in_c*k_size+ic*k_size+k] as f64); }
        }}
        gx[n*in_c*seq_len+ic*seq_len+t] = acc as f32;
    }}}

    // ========== L3 post-zero: 清零 padding 时间步的 grad_x ==========
    for t in 0..seq_len {
        if sequence_mask[t] >= 0.5 {
            // 该时间步是 padding → 所有 batch × channel 位置清零
            for n in 0..batch {
                for ic in 0..in_c {
                    gx[n * in_c * seq_len + ic * seq_len + t] = 0.0;
                }
            }
        }
    }
    // ⚠️ gw 和 gb 不受 L3 mask 影响（padding 只影响 grad_x，不影响参数 grad）

    vec![
        Arc::new(Tensor::<f32, Cpu>::from_data(gx, [batch, in_c, seq_len]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gw, [out_c, in_c, k_size]).unwrap()),
        Arc::new(Tensor::<f32, Cpu>::from_data(gb, [out_c]).unwrap()),
    ]
}
```

---

# 八、sparse_backward 完整过滤流程（架构 0.6.8 稀疏数据流图）

> 本节细化 sparse_backward 如何按 domain_id + active_ids 过滤、如何让未激活算子子图传零梯度、如何自然跳过专家参数的梯度累积。当前文档 §五 有骨架，本节补齐完整执行流程。

## 8.1 完整算法流程

```
输入:
  root_id: NodeId         ← loss 节点（计算图终点）
  active_ids: &[u32]      ← Router 返回的激活领域列表
  nodes: HashMap<NodeId, Node>

输出:
  grads: HashMap<NodeId, Arc<Tensor<f32, Cpu>>>  ← 每个节点的输入梯度

执行流程:
```

```rust
pub fn sparse_backward(
    root_id: NodeId,
    active_ids: &[u32],
    nodes: &HashMap<NodeId, Node>,
) -> HashMap<NodeId, Arc<Tensor<f32, Cpu>>> {
    // ========== Phase 1: 反向拓扑排序（从 root 向上遍历所有可达节点）==========
    let sorted = topo_reverse(nodes, root_id);

    // ========== Phase 2: 初始化 root 梯度 = 全 1 ==========
    let mut grads: HashMap<NodeId, Arc<Tensor<f32, Cpu>>> = HashMap::new();
    if let Some(loss) = nodes.get(&root_id) {
        grads.insert(root_id, Arc::new(Tensor::<f32, Cpu>::ones(
            loss.output.shape().dims.iter().copied()
        )));
    }

    // ========== Phase 3: 反向遍历每个节点 ==========
    for &cid in &sorted {
        let node = match nodes.get(&cid) { Some(n) => n, None => continue };
        let g_out = match grads.get(&cid) {
            Some(g) => g.clone(),
            None => Arc::new(Tensor::<f32, Cpu>::zeros(
                node.output.shape().dims.iter().copied()
            )),
        };

        // ========== 核心稀疏判断（三层过滤）==========
        //
        // Layer 1: 整专家跳过（domain_id ∉ active_ids）
        //   → 属于该领域的所有算子子图一次性跳过
        //   → 零梯度传给所有父节点 → 自然终止该子图的 backward
        //
        // Layer 2: 通道级 mask（Op 枚举内部处理）
        //   → Conv1dMasked / BatchNorm1dMasked 在 op.rs 的具体 backward 分支里处理
        //
        // Layer 3: 序列级 mask（Op 枚举内部处理）
        //   → Conv1dWithSequenceMask 在 op.rs 的具体 backward 分支里处理

        let do_skip_entire_subtree = match node.domain_id {
            Some(did) => !active_ids.contains(&did),
            None => false,  // domain_id=None: 不属于任何专家（TaskHead/Fusion/Loss）→ 正常计算
        };

        if do_skip_entire_subtree {
            // ========== 整子图跳过：给所有父节点传零梯度 ==========
            // 效果：该节点的所有输入节点在后面遍历时，得到的 g_out = zeros
            // 然后它们也会继续把零梯度往上传播 → 自然跳过整个未激活专家子图
            for &pid in &node.inputs {
                if let Some(pn) = nodes.get(&pid) {
                    grads.insert(pid, Arc::new(Tensor::<f32, Cpu>::zeros(
                        pn.output.shape().dims.iter().copied()
                    )));
                }
            }
            continue;  // ⚠️ 关键：跳过这个节点自己的 backward 计算
        }

        // ========== 正常 backward（domain_id=None 或 domain_id∈active_ids）==========
        let inputs: Vec<Arc<Tensor<f32, Cpu>>> = node.inputs.iter()
            .filter_map(|pid| nodes.get(pid).map(|p| p.output.clone()))
            .collect();

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
        for (i, &pid) in node.inputs.iter().enumerate() {
            if i >= input_grads.len() { break; }
            let new_g = input_grads[i].clone();
            match grads.get_mut(&pid) {
                Some(existing) => {
                    // 多个子节点的梯度需要累加（典型情况：一个 Parameter 被多个算子使用）
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
```

## 8.2 自然跳过专家参数梯度累积的机制

```
完整梯度流向（激活专家 domain_id=2）：

  Loss (domain=None)
    │
    ▼ TaskHead (domain=None)
    │
    ▼ Fusion (domain=None)
    │
    ├──▶ Expert domain_id=2 (激活): 所有 Op 正常 backward
    │       └── Parameter grads 正常累积到 SparseParameterPool
    │
    └──▶ Expert domain_id=3 (未激活): do_skip_entire_subtree = true
            └── 给父节点传零梯度 → 所有算子得到 zeros grad_out
                → 但这些算子本身被 continue 跳过 → Parameter.grad 永远不被写
                    → SparseParameterPool 里未激活专家的 Parameter.grad = None
                        → Optimizer.step_sparse 遍历 active_ids 时跳过它们
                            → 完美的端到端稀疏！
```

**关键要点**：
1. Node.domain_id 必须在计算图构建时正确设置（每个专家内部的算子节点都标 `Some(domain_id)`）
2. Param 节点（is_leaf=true, param=Some）必须带 domain_id，否则它们的梯度不会被跳过
3. 梯度写入 SparseParameterPool 的时机是 sparse_backward 返回后，由外部（trainer）统一处理：遍历 `grads` HashMap 找到 param 节点 → 调用 `Parameter.set_grad(grad_tensor)`

---

# 九、四层分层目录规范 + 验收标准

## 9.1 目标目录结构（对齐 00-overview §八）

```
oyta-autograd/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude
│
├── types/                       ← 数据类型层（纯结构体定义）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── node.rs                 ← Node { id, op, inputs, output, is_leaf, param, domain_id } + NodeId
│
├── traits/                      ← 接口层（可选：如果有 trait）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── (预留：如果后续有 BackwardStrategy trait)
│
├── utils/                       ← 工具层（纯算法，不依赖 traits/impl）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── topo_sort.rs            ← topo_reverse：反向拓扑排序（Kahn+BFS）
│
└── impl/                        ← 实现层（Op 枚举 + backward 实现 + sparse_backward 主函数）
    ├── mod.rs                  ← 只 pub mod xxx;
    ├── op.rs                   ← Op 枚举 + compute_backward_op + 三个掩码变体的完整 backward
    ├── backward.rs             ← sparse_backward 主函数 + 域过滤逻辑
    └── engine.rs               ← AutogradEngine：计算图构建 + 管理 + 注册
```

## 9.2 每层职责边界 + mod.rs 铁律

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + utils/ + oyta_tensor | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! impl 层说明

pub mod op;
pub mod backward;
pub mod engine;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::node::{Node, NodeId};
pub use impls::op::{Op, compute_backward_op};
pub use impls::backward::sparse_backward;
pub use impls::engine::AutogradEngine;

pub mod prelude {
    pub use crate::{Node, NodeId, Op, compute_backward_op, sparse_backward, AutogradEngine};
    pub use oyta_tensor::prelude::*;
    pub use std::collections::HashMap;
    pub use std::sync::{Arc, Mutex};
}
```

## 9.3 完整验收标准

```
## 基础验收（原有）
[ ] cargo check --package oyta-autograd 零警告
[ ] Cargo.toml 只写 oyta-tensor = "0.1.1" + rayon（无 rust-version）
[ ] Op 枚举有 4 个掩码变体（Conv1dMasked / BatchNorm1dMasked / Conv1dWithSequenceMask / Router）
[ ] sparse_backward 核心 domain_id 过滤 continue 整子图跳过
[ ] Conv1dMasked backward：grad_out *= (1 - channel_mask[f]) pre-mask → 标准 Conv1d backward
[ ] BatchNorm1dMasked backward：被 mask 特征跳过完整 BN 公式，直接 grad_x[:,f]=0, grad_gamma[f]=0, grad_beta[f]=0
[ ] Conv1dWithSequenceMask backward：标准 Conv1d backward → post-zero padding 时间步 grad_x
[ ] Router Op 返回 Vec::new()
[ ] 每个函数有中文注释

## 追加：sparse_backward 完整流程验收
[ ] topo_reverse 正确实现：从 root 可达的所有节点反向拓扑排序
[ ] 根节点初始化 grad = Tensor::ones（正确的 dL/dloss = 1）
[ ] 未激活专家整子图跳过：do_skip_entire_subtree → 给所有父节点传零梯度 → continue
[ ] domain_id=None 节点（TaskHead/Fusion/Loss）不受过滤影响，正常 backward
[ ] 父节点梯度累加：多个子节点梯度需要 Vec 逐元素加（一个 Parameter 被多个算子引用的情况）
[ ] 零梯度从未激活专家子图自然向上传播到 Parameter → Parameter.grad 永远不被写
[ ] 外部（trainer）遍历 grads HashMap 把 param 节点梯度写回 SparseParameterPool.Parameter.grad

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use
[ ] Node + NodeId 归入 types/ 层
[ ] topo_reverse 纯算法归入 utils/ 层
[ ] Op 枚举 + compute_backward_op + 三个掩码变体 backward + sparse_backward + AutogradEngine 归入 impl/ 层
[ ] 依赖方向单向：types → utils → impls（traits 层预留可空）
```
