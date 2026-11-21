//! oyta-autograd：自动微分引擎（稀疏适配）
//!
//! 四层分层目录结构：
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ lib.rs           顶层统一 pub mod + pub use + prelude               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ types/           数据类型层（纯结构体，零业务逻辑）                  │
//! │   node.rs           Node + NodeId（含 domain_id 稀疏核心字段）      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ traits/          接口层（预留）                                     │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ utils/           工具层（纯算法）                                   │
//! │   topo_sort.rs      topo_reverse（Kahn+BFS 反向拓扑排序）           │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ impls/           实现层（Op 枚举 + backward + engine）              │
//! │   op.rs             Op 枚举 + compute_backward_op + 三个掩码变体    │
//! │   backward.rs       sparse_backward 三层 domain_id 过滤             │
//! │   engine.rs         AutogradEngine 计算图构建管理                    │
//! └─────────────────────────────────────────────────────────────────────┘
//!
//! 稀疏核心设计：
//!   Node 带 domain_id: Option<u32>
//!   sparse_backward 反向遍历时：node.domain_id ∉ active_ids → continue
//!   → 整专家子图跳过 → Parameter.grad 永远不被写 → 完美端到端稀疏

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 公共导出 =====
pub use types::node::{Node, NodeId};
pub use impls::op::{Op, compute_backward_op};
pub use impls::backward::sparse_backward;
pub use impls::engine::AutogradEngine;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{Node, NodeId, Op, compute_backward_op, sparse_backward, AutogradEngine};
    pub use oyta_tensor::prelude::*;
    pub use std::collections::HashMap;
    pub use std::sync::{Arc, Mutex};
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // ===== Op 枚举变体验证 =====
    #[test]
    fn test_op_enum_variants() {
        // 4 个掩码变体必须存在
        let _ = Op::Conv1dMasked { channel_mask: vec![0.5, 0.0, 1.0, 0.0] };
        let _ = Op::BatchNorm1dMasked { channel_mask: vec![0.0, 1.0] };
        let _ = Op::Conv1dWithSequenceMask { sequence_mask: vec![0.0, 0.0, 1.0] };
        let _ = Op::Router { domain_mask: vec![true, false], normalized_weights: vec![0.3, 0.7] };
    }

    // ===== Conv1dMasked backward 基础 shape 验证 =====
    #[test]
    fn test_conv1d_masked_backward_shape() {
        // grad_out: [batch=1, out_c=2, out_len=5]
        let go = Arc::new(Tensor::<f32, Cpu>::zeros([1, 2, 5]));
        // x: [1, in_c=3, seq_len=7]
        let x = Arc::new(Tensor::<f32, Cpu>::ones([1, 3, 7]));
        // w: [2, 3, k_size=3]
        let w = Arc::new(Tensor::<f32, Cpu>::ones([2, 3, 3]));
        let inputs = vec![x, w];
        let channel_mask = vec![0.0f32, 0.0]; // 两个通道都激活

        let grads = crate::impls::op::conv1d_masked_backward(&go, &inputs, &channel_mask);
        assert_eq!(grads.len(), 3); // [gx, gw, gb]
        assert_eq!(grads[0].shape().dims, vec![1, 3, 7]); // gx shape
        assert_eq!(grads[1].shape().dims, vec![2, 3, 3]); // gw shape
        assert_eq!(grads[2].shape().dims, vec![2]);        // gb shape
    }

    // ===== Conv1dMasked pre-mask 效果：被 mask 通道梯度必须为 0 =====
    #[test]
    fn test_conv1d_masked_backward_effective_mask() {
        let go_d = vec![1.0f32; 10]; // 2 * 5
        let go = Arc::new(Tensor::<f32, Cpu>::from_data(go_d, [1, 2, 5]).unwrap());
        let x = Arc::new(Tensor::<f32, Cpu>::ones([1, 1, 5]));
        let w = Arc::new(Tensor::<f32, Cpu>::ones([2, 1, 3]));
        let inputs = vec![x, w];
        // 通道 0 被 mask → channel_mask[0] = 1.0 → inv_mask[0] = 0
        let channel_mask = vec![1.0f32, 0.0];

        let grads = crate::impls::op::conv1d_masked_backward(&go, &inputs, &channel_mask);
        let gb_d = grads[2].data_slice();
        // gb[0] 来自被 mask 通道 → 应为 0
        assert!((gb_d[0]).abs() < 1e-6, "被 mask 通道的 gb 应为 0");
        // gb[1] 来自激活通道 → 不为 0
        assert!((gb_d[1]).abs() > 1e-6, "激活通道的 gb 不应为 0");
    }

    // ===== BatchNorm1dMasked backward shape + mask 效果 =====
    #[test]
    fn test_batchnorm1d_masked_backward() {
        // grad_out: [batch=2, nf=3]
        let go = Arc::new(Tensor::<f32, Cpu>::randn([2, 3]));
        // cached_xhat: [2, 3]，cache 内部提供
        let xhat = Arc::new(Tensor::<f32, Cpu>::ones([2, 3]));
        // cached_inv_std: [3]
        let inv_std = Arc::new(Tensor::<f32, Cpu>::ones([3]));
        // gamma: [3]
        let gamma = Arc::new(Tensor::<f32, Cpu>::ones([3]));
        let inputs = vec![
            Arc::new(Tensor::<f32, Cpu>::zeros([2, 3])), // x_placeholder
            xhat, inv_std, gamma,
        ];
        let channel_mask = vec![0.0f32, 1.0, 0.0]; // 特征 1 被 mask

        let grads = crate::impls::op::batchnorm1d_masked_backward(&go, &inputs, &channel_mask);
        assert_eq!(grads.len(), 3); // [gx, gg, gb]
        assert_eq!(grads[0].shape().dims, vec![2, 3]);
        assert_eq!(grads[1].shape().dims, vec![3]);
        assert_eq!(grads[2].shape().dims, vec![3]);

        // 被 mask 的特征 1 → grad_gamma[1] = 0, grad_beta[1] = 0
        let gg_d = grads[1].data_slice();
        let gb_d = grads[2].data_slice();
        assert!((gg_d[1]).abs() < 1e-6, "被 mask 特征 gg 应为 0");
        assert!((gb_d[1]).abs() < 1e-6, "被 mask 特征 gb 应为 0");
    }

    // ===== Conv1dWithSequenceMask backward shape + post-zero =====
    #[test]
    fn test_conv1d_with_seq_mask_backward() {
        let go = Arc::new(Tensor::<f32, Cpu>::ones([1, 2, 5]));
        let x = Arc::new(Tensor::<f32, Cpu>::ones([1, 2, 7]));
        let w = Arc::new(Tensor::<f32, Cpu>::ones([2, 2, 3]));
        let inputs = vec![x, w];
        // 时间步 5 和 6 是 padding（假设实际 seq_len=5，7-5=2 padding）
        let seq_mask = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let grads = crate::impls::op::conv1d_with_seq_mask_backward(&go, &inputs, &seq_mask);
        assert_eq!(grads.len(), 3);
        let gx_d = grads[0].data_slice();
        // padding 时间步 5,6 的 grad_x 必须全为 0
        let seq_len = 7; let in_c = 2;
        for t in 5..7 {
            for ic in 0..in_c {
                let idx = 0 * in_c * seq_len + ic * seq_len + t;
                assert!((gx_d[idx]).abs() < 1e-6, "padding 时间步 grad_x 应为 0");
            }
        }
    }

    // ===== Router Op backward 返回空 Vec =====
    #[test]
    fn test_router_op_returns_empty_grads() {
        let op = Op::Router {
            domain_mask: vec![true, false],
            normalized_weights: vec![0.6, 0.4],
        };
        let go = Arc::new(Tensor::<f32, Cpu>::ones([2, 4]));
        let inputs: Vec<Arc<Tensor<f32, Cpu>>> = vec![];
        let grads = compute_backward_op(&op, &go, &inputs);
        assert!(grads.is_empty(), "Router Op backward 应返回空 Vec");
    }

    // ===== topo_reverse 基本拓扑 =====
    #[test]
    fn test_topo_reverse_basic() {
        let mut nodes = HashMap::new();
        // 简单图：param0 → Linear → loss
        // param1 ───────↗
        let go = Arc::new(Tensor::<f32, Cpu>::ones([2, 4]));
        nodes.insert(2, Node::new(2, Op::Linear, vec![0, 1], go.clone(), None));
        nodes.insert(0, Node::leaf(0, Tensor::<f32, Cpu>::ones([2, 3]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::ones([3])))), None, true));
        nodes.insert(1, Node::leaf(1, Tensor::<f32, Cpu>::ones([3, 4]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::ones([3, 4])))), None, true));

        let sorted = crate::utils::topo_sort::topo_reverse(&nodes, 2);
        // 反向拓扑顺序：上游 param 先处理，下游 loss 最后处理
        // 正向 Kahn 拓扑 = [loss=2, Linear=None, param0=0, param1=1]？不对...
        // 实际 BFS + Kahn: root=2 → reachable={0,1,2}
        // forward: 0→[2], 1→[2], 2→[]
        // reverse: 2→[0,1], 0→[], 1→[]
        // 入度: deg[0]=1, deg[1]=1, deg[2]=0
        // Kahn: 初始入度 0 = [2] → 处理 2 → deg[0]=0,deg[1]=0 → 队列=[0,1]
        // topo(正向) = [2, 1, 0] → reverse = [0, 1, 2]
        assert_eq!(sorted.len(), 3);
        assert!(sorted.contains(&0));
        assert!(sorted.contains(&1));
        assert_eq!(sorted[2], 2); // loss 最后处理
    }

    // ===== sparse_backward domain_id 过滤核心测试 =====
    #[test]
    fn test_sparse_backward_skips_unactivated_domain() {
        let mut nodes = HashMap::new();
        let zero = |shape| Tensor::<f32, Cpu>::zeros(shape);
        // 构建图：
        // loss(domain=None)
        //   ├── Linear domain_id=1 (激活)
        //   └── Linear domain_id=2 (未激活)
        let loss_out = Arc::new(zero([2, 1]));
        nodes.insert(6, Node::new(6, Op::MSE, vec![4, 5], loss_out, None)); // loss

        let act_out = Arc::new(Tensor::<f32, Cpu>::randn([2, 4]));
        nodes.insert(4, Node::new(4, Op::Linear, vec![0, 1, 2], act_out, Some(1))); // 激活专家

        let skip_out = Arc::new(Tensor::<f32, Cpu>::randn([2, 4]));
        nodes.insert(5, Node::new(5, Op::Linear, vec![0, 3, 2], skip_out, Some(2))); // 未激活专家

        nodes.insert(0, Node::leaf(0, Tensor::<f32, Cpu>::randn([2, 3]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([2, 3])))), None, true));
        nodes.insert(1, Node::leaf(1, Tensor::<f32, Cpu>::randn([3, 4]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([3, 4])))), Some(1), true)); // 激活专家参数
        nodes.insert(2, Node::leaf(2, Tensor::<f32, Cpu>::randn([4]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([4])))), Some(1), true)); // 激活专家参数
        nodes.insert(3, Node::leaf(3, Tensor::<f32, Cpu>::randn([3, 4]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([3, 4])))), Some(2), true)); // 未激活专家参数

        // active_ids = [1]：只有 domain 1 激活
        let active_ids = vec![1u32];
        let grads = sparse_backward(6, &active_ids, &nodes);

        // 激活专家的参数节点（id=1, id=2）应有梯度
        assert!(grads.contains_key(&1), "激活专家参数应在 grads 里");
        assert!(grads.contains_key(&2), "激活专家参数应在 grads 里");

        // 未激活专家的参数节点（id=3）应有零梯度（被 domain_id=2 过滤 → 父节点传零）
        if let Some(g3) = grads.get(&3) {
            let d3 = g3.data_slice();
            assert!(d3.iter().all(|&v| v == 0.0), "未激活专家参数梯度应为全 0");
        }
    }

    // ===== domain_id=None 节点不受过滤 =====
    #[test]
    fn test_domain_id_none_not_filtered() {
        let mut nodes = HashMap::new();
        // 图：TaskHead(domain=None) → loss
        let loss_out = Arc::new(Tensor::<f32, Cpu>::ones([2, 1]));
        nodes.insert(2, Node::new(2, Op::MSE, vec![1], loss_out, None)); // loss
        let head_out = Arc::new(Tensor::<f32, Cpu>::randn([2, 3]));
        nodes.insert(1, Node::new(1, Op::Linear, vec![0, 99], head_out, None)); // domain=None → 不受过滤
        nodes.insert(0, Node::leaf(0, Tensor::<f32, Cpu>::randn([2, 4]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([2, 4])))), None, false)); // 输入数据不需要梯度
        nodes.insert(99, Node::leaf(99, Tensor::<f32, Cpu>::randn([4, 3]),
            Some(Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([4, 3])))), None, true)); // TaskHead 参数

        // 空 active_ids → 所有 domain_id=Some(did) 的专家都被跳过
        // 但 domain_id=None 的 TaskHead/Loss 正常计算
        let active_ids = vec![100u32]; // 没有 domain 100
        let grads = sparse_backward(2, &active_ids, &nodes);
        // domain_id=None 的节点应正常 backward → grads 里有 0 和 99
        assert!(grads.contains_key(&0), "domain_id=None 叶子应有梯度");
    }

    // ===== AutogradEngine 注册 + backward =====
    #[test]
    fn test_autograd_engine_basic() {
        let mut eng = AutogradEngine::new();
        assert_eq!(eng.node_count(), 0);

        let x_param = Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([2, 3])));
        let x_id = eng.register_leaf(
            Tensor::<f32, Cpu>::randn([2, 3]), Some(x_param), None, false,
        ); // 输入数据不需要梯度

        let w_param = Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([3, 4])));
        let w_id = eng.register_leaf(
            Tensor::<f32, Cpu>::randn([3, 4]), Some(w_param), Some(1), true,
        );

        let b_param = Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([4])));
        let b_id = eng.register_leaf(
            Tensor::<f32, Cpu>::randn([4]), Some(b_param), Some(1), true,
        );

        let act = Tensor::<f32, Cpu>::randn([2, 4]);
        let act_id = eng.register_node(
            Op::Linear, vec![x_id, w_id, b_id],
            Arc::new(act), Some(1),
        );

        let loss = Arc::new(Tensor::<f32, Cpu>::randn([2, 1]));
        let loss_id = eng.register_node(Op::MSE, vec![act_id], loss, None);

        assert_eq!(eng.node_count(), 5);

        // 执行 sparse_backward（active_ids=[1] → 激活专家正常计算）
        let grads = eng.backward(loss_id, &[1]);
        assert!(grads.contains_key(&act_id));
        assert!(grads.contains_key(&x_id));
    }

    // ===== 清空 Engine =====
    #[test]
    fn test_autograd_engine_clear() {
        let mut eng = AutogradEngine::new();
        let p = Arc::new(Mutex::new(Tensor::<f32, Cpu>::randn([3])));
        eng.register_leaf(Tensor::<f32, Cpu>::randn([3]), Some(p), None, true);
        assert_eq!(eng.node_count(), 1);
        eng.clear();
        assert_eq!(eng.node_count(), 0);
    }
}
