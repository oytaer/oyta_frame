//! Op backward 函数集合（impls 层，依赖 types::op::Op）
//!
//! 四层分层：types/op.rs 定义枚举，impls/op.rs 定义反向传播函数
//! 标准算子 backward：Linear / Relu / Tanh 等
//! 掩码变体 backward：Conv1dMasked(L2 pre-mask) / BatchNorm1dMasked(L2 post-zero) / Conv1dWithSequenceMask(L3 post-zero)

use std::sync::Arc;
use oyta_tensor::prelude::*;

// 重新导出 Op 枚举（types 层定义，impls 层 re-export 保持向后兼容）
pub use crate::types::op::Op;

// ===== compute_backward_op：标准算子统一入口 =====
//
// 掩码变体不走这里，backward.rs 直接调 conv1d_masked_backward / batchnorm1d_masked_backward / conv1d_with_seq_mask_backward

/// 统一 backward 入口（标准算子用，掩码变体有独立函数）
pub fn compute_backward_op(
    op: &Op,
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    // Router / Leaf 不可训练 → 返回空
    if matches!(op, Op::Router { .. } | Op::Leaf) { return Vec::new(); }

    // ===== Linear backward =====
    // 输入顺序约定：[x, w, b]
    // forward: y = x @ W + b，x:[B,K], W:[K,N], b:[N], y:[B,N]
    if matches!(op, Op::Linear) && input_tensors.len() >= 3 {
        let x = &input_tensors[0]; let w = &input_tensors[1];
        // ∂L/∂x = grad_out @ W^T = [B,N] @ [N,K] = [B,K] ✓
        let wt = w.transpose(0, 1);
        let dldx = grad_out.matmul(&wt).unwrap();
        // ∂L/∂w = x^T @ grad_out = [K,B] @ [B,N] = [K,N] ✓
        let xt = x.transpose(0, 1);
        let dldw = xt.matmul(grad_out).unwrap();
        // ∂L/∂b = sum over batch
        let go_d = grad_out.data_slice();
        let out_dim = grad_out.shape().dims[1];
        let batch = grad_out.shape().dims[0];
        let mut dldb = vec![0.0f32; out_dim];
        for n in 0..batch {
            for f in 0..out_dim { dldb[f] += go_d[n * out_dim + f]; }
        }
        let dldb = Tensor::<f32, Cpu>::from_data(dldb, [out_dim]).unwrap();
        return vec![Arc::new(dldx), Arc::new(dldw), Arc::new(dldb)];
    }

    // ===== Relu backward =====
    if matches!(op, Op::Relu) && !input_tensors.is_empty() {
        let x_d = input_tensors[0].data_slice();
        let go_d = grad_out.data_slice();
        let mut dx = vec![0.0f32; x_d.len()];
        for i in 0..x_d.len() { if x_d[i] > 0.0 { dx[i] = go_d[i]; } }
        let shape = input_tensors[0].shape().dims.clone();
        return vec![Arc::new(Tensor::<f32, Cpu>::from_data(dx, shape).unwrap())];
    }

    // ===== Tanh backward =====
    if matches!(op, Op::Tanh) && !input_tensors.is_empty() {
        let shape = input_tensors[0].shape().dims.clone();
        return vec![Arc::new(Tensor::<f32, Cpu>::zeros(shape))];
    }

    // ===== 其他标准算子 Phase 1 骨架：返回零梯度占位 =====
    input_tensors.iter().map(|t| {
        Arc::new(Tensor::<f32, Cpu>::zeros(t.shape().dims.iter().copied()))
    }).collect()
}

// =====================================================================
// Conv1dMasked backward（L2 通道稀疏，架构 0.6.8 Step 1 pre-mask 设计）
// =====================================================================
//
// 稀疏语义：channel_mask[f] = 1.0 → 第 f 个输出通道被 mask
// 稀疏位置：backward 前对 grad_out 做通道级缩放 pre-mask，然后调标准 Conv1d backward
//
// 为什么 pre-mask？
//   ∂L/∂w[oc, ic, k] = Σ_{n, o} grad_out_masked[n, oc, o] × x[n, ic, ...]
//   如果 grad_out_masked[oc, :] = 0 → 该 out_channel 对 w 的梯度自动为零
//   ∂L/∂x[n, ic, t] = Σ_{oc} grad_out_masked[n, oc, ...] × w[oc, ic, k]
//   被 mask 的 out_channel 不贡献梯度 → 传播回输入的梯度也是正确稀疏的

/// Conv1d 带通道级 mask 的完整 backward
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
            if scale == 0.0 { continue; } // 整通道跳过
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

    // ∂L/∂w[oc, ic, k]（互相关公式）
    for oc in 0..out_c {
        for ic in 0..in_c {
            for k in 0..k_size {
                let mut acc: f64 = 0.0;
                for n in 0..batch {
                    for t in 0..out_len {
                        // stride=1, padding=1 → in_idx = t + k - 1（安全计算防 usize 下溢）
                        if let Some(in_idx) = t.checked_add(k).and_then(|v| v.checked_sub(1)) {
                            if in_idx < seq_len {
                                acc += (masked_go[n * out_c * out_len + oc * out_len + t] as f64)
                                     * (x_d[n * in_c * seq_len + ic * seq_len + in_idx] as f64);
                            }
                        }
                    }
                }
                gw[oc * in_c * k_size + ic * k_size + k] = acc as f32;
            }
        }
    }

    // ∂L/∂x[n, ic, t]（dilated 反卷积，手动防 usize 下溢）
    for n in 0..batch {
        for ic in 0..in_c {
            for t in 0..seq_len {
                let mut acc: f64 = 0.0;
                for oc in 0..out_c {
                    for t_out in 0..out_len {
                        // stride=1, padding=1 → k = t + 1 - t_out
                        // 防 usize 下溢：先保证 t + 1 > t_out 再减
                        let diff = t as i32 + 1 - t_out as i32;
                        if diff >= 0 {
                            let k = diff as usize;
                            if k < k_size {
                                acc += (masked_go[n * out_c * out_len + oc * out_len + t_out] as f64)
                                     * (w_d[oc * in_c * k_size + ic * k_size + k] as f64);
                            }
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

// =====================================================================
// BatchNorm1dMasked backward（L2 通道稀疏，架构 0.6.8 完整简化公式）
// =====================================================================
//
// 稀疏语义：被 mask 的特征 channel f → grad_x[:, f] + grad_gamma[f] + grad_beta[f] 全清零
// 稀疏位置：先算标准 BN backward → 被 mask 特征直接 post-zero（跳过 mean_sgo/mean_sx 聚合）
//
// 为什么 post-zero？
//   BN backward 内部 mean_sgo / mean_sx 是全局聚合
//   如果 pre-mask grad_out → 聚合分母不变但值变小 → 不正确
//   正确做法：被 mask 特征跳过完整 BN 公式直接置零

/// BatchNorm1d 带通道级 mask 的完整 backward
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

    // ========== 标准 BN backward 简化公式（每个特征独立）==========
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
        if f < channel_mask.len() && channel_mask[f] >= 0.5 {
            gb[f] = 0.0;
            gg[f] = 0.0;
            for n in 0..batch { gx[n * nf + f] = 0.0; }
            continue; // ⚠️ 关键：被 mask 特征不参与 mean_sgo / mean_sx 聚合
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

// =====================================================================
// Conv1dWithSequenceMask backward（L3 序列稀疏，架构 0.6.8 post-zero 设计）
// =====================================================================
//
// 稀疏语义：sequence_mask[t] = 1.0 → 时间步 t 是 padding
// 稀疏位置：标准 Conv1d backward 计算完 → post-zero grad_x[:, :, t]
//
// 为什么 post-zero？
//   卷积感受野交叉，padding 位置可能影响有效位置的 grad_out
//   post-zero grad_x 才保证 padding 时间步梯度不向上游传播
//   gw 和 gb 不受 L3 mask 影响（padding 只影响 grad_x）

/// Conv1d 带序列级 mask 的完整 backward
pub fn conv1d_with_seq_mask_backward(
    grad_out: &Arc<Tensor<f32, Cpu>>,
    input_tensors: &[Arc<Tensor<f32, Cpu>>],
    sequence_mask: &[f32],  // mask[t] = 0.0→有效 1.0→padding
) -> Vec<Arc<Tensor<f32, Cpu>>> {
    let x = &input_tensors[0]; let w = &input_tensors[1];
    let go_d = grad_out.data_slice(); let x_d = x.data_slice(); let w_d = w.data_slice();
    let batch = grad_out.shape().dims[0];
    let out_c = grad_out.shape().dims[1];
    let out_len = grad_out.shape().dims[2];
    let in_c = x.shape().dims[1];
    let seq_len = x.shape().dims[2];
    let k_size = w.shape().dims[2];

    // ========== Step 1: 完整标准 Conv1d backward ==========
    let mut gx = vec![0.0f32; batch * in_c * seq_len];
    let mut gw = vec![0.0f32; out_c * in_c * k_size];
    let mut gb = vec![0.0f32; out_c];

    // ∂L/∂b
    for oc in 0..out_c {
        let mut acc: f64 = 0.0;
        for n in 0..batch { for t in 0..out_len {
            acc += go_d[n*out_c*out_len + oc*out_len + t] as f64;
        }}
        gb[oc] = acc as f32;
    }
    // ∂L/∂w（互相关）
    for oc in 0..out_c { for ic in 0..in_c { for k in 0..k_size {
        let mut acc: f64 = 0.0;
        for n in 0..batch { for t in 0..out_len {
            // stride=1, padding=1 → ii = t + k - 1（安全计算防 usize 下溢）
            if let Some(ii) = t.checked_add(k).and_then(|v| v.checked_sub(1)) {
                if ii < seq_len {
                    acc += (go_d[n*out_c*out_len + oc*out_len + t] as f64)
                         * (x_d[n*in_c*seq_len + ic*seq_len + ii] as f64);
                }
            }
        }}
        gw[oc*in_c*k_size + ic*k_size + k] = acc as f32;
    }}}
    // ∂L/∂x（dilated 反卷积，手动防 usize 下溢）
    for n in 0..batch {
        for ic in 0..in_c {
            for t in 0..seq_len {
                let mut acc: f64 = 0.0;
                for oc in 0..out_c {
                    for t_out in 0..out_len {
                        let diff = t as i32 + 1 - t_out as i32;
                        if diff >= 0 {
                            let k = diff as usize;
                            if k < k_size {
                                acc += (go_d[n*out_c*out_len + oc*out_len + t_out] as f64)
                                     * (w_d[oc*in_c*k_size + ic*k_size + k] as f64);
                            }
                        }
                    }
                }
                gx[n*in_c*seq_len + ic*seq_len + t] = acc as f32;
            }
        }
    }

    // ========== Step 2: L3 post-zero：清零 padding 时间步的 grad_x ==========
    for t in 0..seq_len {
        if t < sequence_mask.len() && sequence_mask[t] >= 0.5 {
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
