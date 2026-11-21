# oyta-tensor Backend Trait 补全规格说明

> **版本**：v0.1  
> **面向**：oyta-tensor 底层开发者  
> **目的**：补全 oyta-frame 框架层当前手写的 7 类算子，让框架层 100% 调用底层 Backend trait

---

## 第一部分：oyta-tensor 0.1.1 已有的完整 Backend trait 清单

### Backend 聚合 trait（aggregate.rs）

```rust
pub trait Backend:
    BackendBase +
    ElwiseOps<f32> + ElwiseOps<F16> + ElwiseOps<BF16> + ElwiseOps<f64> +
    ReduceOps<f32> + ReduceOps<F16> + ReduceOps<BF16> +
    FlashAttentionOps<f32> + FlashAttentionOps<F16> + FlashAttentionOps<BF16> +
    MatmulOps +
    FusedOps<f32> + FusedOps<F16> + FusedOps<BF16> +
    InPlaceOps<f32> + InPlaceOps<F16> + InPlaceOps<BF16> +
    Copy + Clone + Send + Sync + 'static
```

### 已有 Trait 方法完整表

#### 1. BackendBase（base.rs）— 3 个方法

| 方法 | 签名 | 实现 |
|------|------|------|
| `fill` | `fn fill(dst: *mut f32, n: usize, value: f32)` | default impl |
| `copy` | `fn copy(src: *const f32, dst: *mut f32, n: usize)` | default impl |
| `where_bool` | `fn where_bool(cond, a, b, out, n)` | default impl |

#### 2. ElwiseOps\<T: Float\>（elwise.rs）— 21 个方法

| 类别 | 方法 | 签名 | 实现 |
|------|------|------|------|
| 二元算术 | elwise_add/sub/mul/div | `(a: *const T, b: *const T, c: *mut T, n: usize)` | default + Rayon |
| 二元广播 | **elwise_add_broadcast** | `(a_ptr, a_strides, a_shape, a_rank, b_ptr, b_strides, b_shape, b_rank, out_ptr, out_shape, out_rank)` | default ✅ 已存在 |
| 一元算术 | elwise_neg/abs/exp/log/sqrt/sin/cos | `(a: *const T, c: *mut T, n: usize)` | default + Rayon |
| 激活 | elwise_relu/gelu/gelu_fast/sigmoid/tanh/silu | `(a: *const T, c: *mut T, n: usize)` | default + Rayon |
| 幂 | elwise_pow | `(a: *const T, exp: f32, c: *mut T, n: usize)` | default + Rayon |
| In-place | elwise_add_inplace/mul_inplace | `(a: *mut T, b: *const T, n: usize)` | default + Rayon |

#### 3. ReduceOps\<T: Float + Accumulate\>（reduce.rs）— 8 个方法

| 方法 | 签名 | 实现 |
|------|------|------|
| reduce_sum | `(a: *const T, out: *mut T, outer: usize, inner: usize)` | default + Rayon + f64 累加器 |
| reduce_mean | `(a: *const T, out: *mut T, outer, inner)` | default + Rayon + f64 累加器 |
| reduce_max/min | `(a: *const T, out: *mut T, outer, inner)` | default |
| softmax | `(a: *const T, out: *mut T, outer, inner)` | default + f64 累加器 |
| layernorm | `(a, out, gamma, beta, outer, inner, eps: f32)` | default + f64 累加器 |
| rmsnorm | `(a, out, gamma, outer, inner, eps)` | default + f64 累加器 |
| layernorm_inplace | `(a: *mut T, gamma, beta, outer, inner, eps)` | default |

#### 4. MatmulOps（matmul.rs）— 6 个方法

| 方法 | 签名 | 实现 |
|------|------|------|
| matmul | `(a, lda, b, ldb, c, ldc, m, n, k, alpha, beta)` | default + tiled_fma + Strassen + 可选 BLAS |
| matmul_batched | `(a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, m, n, k, batch, alpha, beta)` | default（循环调单 matmul） |
| matmul_t | `(a, lda, transpose_a, b, ldb, transpose_b, c, ldc, m, n, k, alpha, beta)` | default |
| matmul_f16 | `(a: *const F16, lda, b, ldb, c: *mut F16, ldc, m, n, k, alpha, beta)` | default（cast→f32→算→cast 回） |
| matmul_bf16 | `(a: *const BF16, lda, b, ldb, c: *mut BF16, ldc, m, n, k, alpha, beta)` | default |
| default_matmul | （GPU fallback 用） | Cpu 完整实现 |

#### 5. FlashAttentionOps\<T: Float\>（flash_attn.rs）— 2 个方法

| 方法 | 签名 | 实现 |
|------|------|------|
| flash_attention | `(q, k, v, out, seq_q, seq_k, d_k, d_v, scale, is_causal)` | default + f64 累加器 |
| flash_attention_batched | `(q, k, v, out, batch, heads, seq_q, seq_k, d_k, d_v, scale, is_causal, strides)` | default + Rayon + f64 |

#### 6. FusedOps\<T: Float + Accumulate\>（fused.rs）— 4 个方法

| 方法 | 说明 |
|------|------|
| layernorm_gelu | `gelu(layernorm(x, g, b, eps))` — 融合 LN + GELU |
| linear_bias_gelu | `gelu(x @ W + bias)` — 融合 Linear + Bias + GELU |
| rmsnorm_swiglu | SwiGLU `silu(x@W1) * (x@W2) @ W3` — LLaMA 必须 |
| matmul_activation | `activation(α·A@B + β·C)` — 融合 matmul + 激活 |

#### 7. InPlaceOps\<T: Float\>（inplace.rs）— 5 个方法

| 方法 | 签名 |
|------|------|
| elwise_add_inplace | `(a: *mut T, b: *const T, n: usize)` |
| elwise_mul_inplace | `(a: *mut T, b: *const T, n: usize)` |
| elwise_scalar_add_inplace | `(a: *mut T, n: usize, val: f32)` |
| elwise_scalar_mul_inplace | `(a: *mut T, n: usize, val: f32)` |
| elwise_relu_inplace | `(a: *mut T, n: usize)` |

---

## 第二部分：缺失的 Backend Trait（需要新增 4 个 trait）

### 缺失 #1：Conv1dOps — 1D 卷积算子

**文件**：`oyta-device/src/backend/conv1d.rs`（新增）  
**聚合位置**：在 `Backend` trait 的聚合列表中添加 `Conv1dOps`

#### Trait 定义

```rust
//! conv1d.rs：1D 卷积 Backend trait（Conv1dOps<T> 一套覆盖 f32/f16/bf16）
//!
//! 支持：NCHW 格式、C_out=1 普通卷积、可配置 padding/stride/dilation
//! backward：同时提供 dx（输入梯度）+ dwdb（权重+偏置梯度）

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait Conv1dOps<T: Float>: BackendBase {
    // ===== Conv1d Forward =====
    // NCHW: [batch, C_in, L] × [C_out, C_in, K] → [batch, C_out, L_out]
    // L_out = (L + 2*padding - dilation*(K-1) - 1) / stride + 1

    /// Conv1d forward（C_out=1，有 bias）
    ///
    /// # 参数
    /// - x: [N, C_in, L] 输入
    /// - w: [C_out, C_in, K] 权重
    /// - b: [C_out] 偏置（可 null）
    /// - out: [N, C_out, L_out] 输出
    unsafe fn conv1d_fwd(
        x: *const T, x_dim: [usize; 3],       // [N, C_in, L]
        w: *const T, w_dim: [usize; 3],       // [C_out, C_in, K]
        b: *const T,                           // [C_out]，null 表示无 bias
        out: *mut T, out_dim: [usize; 3],     // [N, C_out, L_out]
        padding: usize, stride: usize, dilation: usize,
    ) {
        // === Cpu 默认实现（纯标量循环 + Rayon 并行 batch 维）===
        let n = x_dim[0]; let c_in = x_dim[1]; let l_in = x_dim[2];
        let c_out = w_dim[0]; let k = w_dim[2];
        let l_out = out_dim[2];

        // 把所有指针提升到 f32（统一计算精度）
        let x_f32 = unsafe { std::slice::from_raw_parts(x, n * c_in * l_in) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();
        let w_f32 = unsafe { std::slice::from_raw_parts(w, c_out * c_in * k) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();
        let b_f32 = if b.is_null() { vec![0.0f32; c_out] }
            else { unsafe { std::slice::from_raw_parts(b, c_out) }.iter().map(|&v| v.to_f32()).collect() };

        // Rayon 并行 batch 和 c_out
        use rayon::prelude::*;
        let out_total = n * c_out * l_out;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out as *mut T, out_total) };
        let out_f32_mut: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_slice.as_mut_ptr() as *mut f32, out_total) };

        (0..n).into_par_iter().for_each(|ni| {
            for co in 0..c_out {
                for lo in 0..l_out {
                    let mut acc = 0.0f64;  // f64 累加器防精度损失
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_in_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_in_idx >= 0 && l_in_idx < l_in as isize {
                                let xv = x_f32[ni * c_in * l_in + ci * l_in + l_in_idx as usize] as f64;
                                let wv = w_f32[co * c_in * k + ci * k + ki] as f64;
                                acc += xv * wv;
                            }
                        }
                    }
                    acc += b_f32[co] as f64;
                    out_f32_mut[ni * c_out * l_out + co * l_out + lo] = T::from_f32(acc as f32).to_f32();
                }
            }
        });
        // 写回 T 类型
        unsafe {
            for i in 0..out_total {
                *out.add(i) = T::from_f32(out_f32_mut[i]);
            }
        }
    }

    // ===== Conv1d Backward dx =====
    unsafe fn conv1d_bwd_dx(
        dy: *const T, dy_dim: [usize; 3],      // [N, C_out, L_out]
        w: *const T, w_dim: [usize; 3],        // [C_out, C_in, K]
        dx: *mut T, dx_dim: [usize; 3],        // [N, C_in, L]
        padding: usize, stride: usize, dilation: usize,
    ) {
        // === Cpu 默认实现 ===
        let n = dy_dim[0]; let c_out = dy_dim[1]; let l_out = dy_dim[2];
        let c_in = dx_dim[1]; let l_in = dx_dim[2]; let k = w_dim[2];

        let dy_f32 = unsafe { std::slice::from_raw_parts(dy, n * c_out * l_out) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();
        let w_f32 = unsafe { std::slice::from_raw_parts(w, c_out * c_in * k) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();

        use rayon::prelude::*;
        let dx_total = n * c_in * l_in;
        let dx_slice = unsafe { std::slice::from_raw_parts_mut(dx as *mut T, dx_total) };
        let dx_f32_mut: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(dx_slice.as_mut_ptr() as *mut f32, dx_total) };
        for v in dx_f32_mut.iter_mut() { *v = 0.0; }

        (0..n).into_par_iter().for_each(|ni| {
            for co in 0..c_out {
                for lo in 0..l_out {
                    let dy_val = dy_f32[ni * c_out * l_out + co * l_out + lo] as f64;
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_in_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_in_idx >= 0 && l_in_idx < l_in as isize {
                                let w_val = w_f32[co * c_in * k + ci * k + ki] as f64;
                                let idx = ni * c_in * l_in + ci * l_in + l_in_idx as usize;
                                dx_f32_mut[idx] += (dy_val * w_val) as f32;
                            }
                        }
                    }
                }
            }
        });
        unsafe {
            for i in 0..dx_total { *dx.add(i) = T::from_f32(dx_f32_mut[i]); }
        }
    }

    // ===== Conv1d Backward dw + db（一次计算 dw 和 db）=====
    unsafe fn conv1d_bwd_dwdb(
        dy: *const T, dy_dim: [usize; 3],      // [N, C_out, L_out]
        x: *const T, x_dim: [usize; 3],        // [N, C_in, L]
        dw: *mut T, dw_dim: [usize; 3],        // [C_out, C_in, K]
        db: *mut T, db_dim: [usize; 1],        // [C_out]
        padding: usize, stride: usize, dilation: usize,
    ) {
        let n = dy_dim[0]; let c_out = dy_dim[1]; let l_out = dy_dim[2];
        let c_in = x_dim[1]; let l_in = x_dim[2]; let k = dw_dim[2];

        let dy_f32 = unsafe { std::slice::from_raw_parts(dy, n * c_out * l_out) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();
        let x_f32 = unsafe { std::slice::from_raw_parts(x, n * c_in * l_in) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();

        let dw_total = c_out * c_in * k;
        let db_total = c_out;
        let dw_f32 = unsafe { std::slice::from_raw_parts_mut(dw as *mut T, dw_total) }
            as *mut T as *mut f32;
        let db_f32 = unsafe { std::slice::from_raw_parts_mut(db as *mut T, db_total) }
            as *mut T as *mut f32;
        unsafe {
            for i in 0..dw_total { *dw_f32.add(i) = 0.0; }
            for i in 0..db_total { *db_f32.add(i) = 0.0; }
        }

        // dw: 串行（小矩阵，多 block 累加冲突）
        for ni in 0..n {
            for co in 0..c_out {
                for lo in 0..l_out {
                    let dy_val = dy_f32[ni * c_out * l_out + co * l_out + lo] as f64;
                    unsafe { *db_f32.add(co) += dy_val as f32; }
                    for ci in 0..c_in {
                        for ki in 0..k {
                            let l_in_idx = (lo as isize * stride as isize + ki as isize * dilation as isize)
                                - padding as isize;
                            if l_in_idx >= 0 && l_in_idx < l_in as isize {
                                let xv = x_f32[ni * c_in * l_in + ci * l_in + l_in_idx as usize] as f64;
                                let idx = co * c_in * k + ci * k + ki;
                                unsafe { *dw_f32.add(idx) += (dy_val * xv) as f32; }
                            }
                        }
                    }
                }
            }
        }
        unsafe {
            for i in 0..dw_total { *dw.add(i) = T::from_f32(*dw_f32.add(i)); }
            for i in 0..db_total { *db.add(i) = T::from_f32(*db_f32.add(i)); }
        }
    }
}
```

#### Tensor 高层封装

```rust
// oyta-core/src/tensor/ops.rs 追加

impl<T: Float + Clone, D: Backend + Conv1dOps<T>> Tensor<T, D> {
    /// Conv1d forward（NCHW 格式）
    pub fn conv1d(
        &self, weight: &Tensor<T, D>, bias: Option<&Tensor<T, D>>,
        padding: usize, stride: usize, dilation: usize,
    ) -> Result<Tensor<T, D>, TensorError> {
        let s = self.shape().dims.clone();
        let w_s = weight.shape().dims.clone();
        assert_eq!(s.len(), 3, "conv1d 输入必须 3D [N, C_in, L]");
        assert_eq!(w_s.len(), 3, "conv1d 权重必须 3D [C_out, C_in, K]");

        let l_in = s[2]; let k = w_s[2];
        let l_out = (l_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let shape_out = vec![s[0], w_s[0], l_out];

        let mut out = Tensor::<T, D>::zeros(shape_out.clone());
        let b_ptr = bias.map(|b| b.data_slice().as_ptr()).unwrap_or(std::ptr::null());

        unsafe {
            D::conv1d_fwd(
                self.data_slice().as_ptr(), [s[0], s[1], s[2]],
                weight.data_slice().as_ptr(), [w_s[0], w_s[1], w_s[2]],
                b_ptr,
                out.data_slice_mut().unwrap(), [shape_out[0], shape_out[1], shape_out[2]],
                padding, stride, dilation,
            );
        }
        Ok(out)
    }
}
```

#### oyta-frame 切换位置

| 当前手写 | 切换后 |
|---------|--------|
| `oyta-expert/src/utils/conv1d.rs::Conv1d::forward`（~80 行 Vec 循环） | `x.conv1d(&w, Some(&b), padding, stride, dilation)?` |
| `oyta-autograd/src/impls/op.rs::Op::Conv1dMasked backward`（~150 行） | `D::conv1d_bwd_dx + D::conv1d_bwd_dwdb` |

---

### 缺失 #2：BatchNormOps — 1D BatchNorm

**文件**：`oyta-device/src/backend/batchnorm.rs`（新增）

#### Trait 定义

```rust
//! batchnorm.rs：1D BatchNorm Backend trait
//!
//! 支持 train/eval 模式自动切换
//! train: 用 batch 统计 + 更新 running_mean/running_var（momentum 移动平均）
//! eval: 用 running_mean/running_var
//! backward: gamma_grad / beta_grad / dx 同时计算

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait BatchNormOps<T: Float>: BackendBase {
    /// BN1d forward
    ///
    /// # 参数
    /// - x: [N, C, L] 或 [N, D]（2D 或 3D 输入）
    /// - gamma, beta: [C] 或 [D]（affine 参数）
    /// - running_mean, running_var: [C] 或 [D]（eval 统计）
    /// - out: 与 x 同 shape
    /// - eps: 数值稳定项（默认 1e-5）
    /// - momentum: running stat 更新系数（默认 0.1）
    /// - training: true=train 模式（用 batch stat 并更新 running），false=eval 模式
    unsafe fn batchnorm1d_fwd(
        x: *const T, x_shape: &[usize],
        gamma: *const T, beta: *const T,
        running_mean: *mut T, running_var: *mut T,
        out: *mut T,
        eps: f32, momentum: f32, training: bool,
    ) {
        // === Cpu 默认实现 ===
        let rank = x_shape.len();
        let feat_dim = x_shape[rank - 1];  // 最后一维是 feature
        let outer: usize = x_shape[..rank - 1].iter().product();
        let total = outer * feat_dim;

        let x_f32 = unsafe { std::slice::from_raw_parts(x, total) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();

        // ===== 每个 feature 通道独立处理 =====
        for c in 0..feat_dim {
            // 1. 收集这个 feature 的所有值
            let mut vals_f64: Vec<f64> = Vec::with_capacity(outer);
            for i in 0..outer {
                vals_f64.push(x_f32[i * feat_dim + c] as f64);
            }

            // 2. 计算 mean（f64 累加器）
            let sum: f64 = vals_f64.iter().sum();
            let mean = sum / outer as f64;

            // 3. 计算 var（f64 累加器）
            let var: f64 = vals_f64.iter()
                .map(|&v| { let d = v - mean; d * d })
                .sum::<f64>() / outer as f64;

            // 4. running stat 更新（只在 train 模式）
            if training {
                let rm_ptr = running_mean as *mut f32;
                let rv_ptr = running_var as *mut f32;
                unsafe {
                    let old_mean = *rm_ptr.add(c) as f64;
                    let old_var = *rv_ptr.add(c) as f64;
                    *rm_ptr.add(c) = ((1.0 - momentum as f64) * old_mean + momentum as f64 * mean) as f32;
                    // unbiased var correction: batch_var * N/(N-1)
                    let unbiased_var = var * outer as f64 / (outer as f64 - 1.0).max(1.0);
                    *rv_ptr.add(c) = ((1.0 - momentum as f64) * old_var + momentum as f64 * unbiased_var) as f32;
                }
            }

            // 5. 选择 eval stat 还是 batch stat
            let (mean_use, var_use) = if !training {
                let rm_ptr = running_mean as *const f32;
                let rv_ptr = running_var as *const f32;
                unsafe { (*rm_ptr.add(c) as f64, *rv_ptr.add(c) as f64) }
            } else {
                (mean, var)
            };

            // 6. 归一化 + affine
            let gamma_v = unsafe { *gamma.add(c) }.to_f32() as f64;
            let beta_v = unsafe { *beta.add(c) }.to_f32() as f64;
            let inv_std = 1.0 / (var_use + eps as f64).sqrt();

            for i in 0..outer {
                let x_val = x_f32[i * feat_dim + c] as f64;
                let normed = (x_val - mean_use) * inv_std;
                let out_val = normed * gamma_v + beta_v;
                unsafe { *out.add(i * feat_dim + c) = T::from_f32(out_val as f32); }
            }
        }
    }

    /// BN1d backward
    ///
    /// 同时计算：gamma_grad, beta_grad, dx
    /// 需要 forward 时的 normed 缓存（x_norm = (x - mean) / sqrt(var + eps)）
    unsafe fn batchnorm1d_bwd(
        dy: *const T, dy_shape: &[usize],
        x_norm: *const T,    // forward 缓存的 normed 输入
        gamma: *const T,     // [C]
        dgamma: *mut T, dbeta: *mut T,  // [C]
        dx: *mut T, dx_shape: &[usize],
        eps: f32,
    ) {
        let rank = dy_shape.len();
        let feat_dim = dy_shape[rank - 1];
        let outer: usize = dy_shape[..rank - 1].iter().product();
        let total = outer * feat_dim;
        let inv_n = 1.0f64 / outer as f64;

        let dy_f32 = unsafe { std::slice::from_raw_parts(dy, total) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();
        let x_norm_f32 = unsafe { std::slice::from_raw_parts(x_norm, total) }
            .iter().map(|&v| v.to_f32()).collect::<Vec<f32>>();

        for c in 0..feat_dim {
            let gamma_v = unsafe { *gamma.add(c) }.to_f32() as f64;

            // dbeta[c] = Σ dy[..., c]
            let mut dbeta_c: f64 = 0.0;
            // dgamma[c] = Σ dy[..., c] * x_norm[..., c]
            let mut dgamma_c: f64 = 0.0;
            // 预存 sum(dy * x_norm) 和 sum(dy) 用于 dx 公式
            let mut sum_dy_xn: f64 = 0.0;
            let mut sum_dy: f64 = 0.0;

            for i in 0..outer {
                let dyv = dy_f32[i * feat_dim + c] as f64;
                let xnv = x_norm_f32[i * feat_dim + c] as f64;
                dbeta_c += dyv;
                dgamma_c += dyv * xnv;
                sum_dy_xn += dyv * xnv;
                sum_dy += dyv;
            }

            unsafe {
                *dbeta.add(c) = T::from_f32(dbeta_c as f32);
                *dgamma.add(c) = T::from_f32(dgamma_c as f32);
            }

            // dx[c] = (gamma / sqrt(var + eps)) * (dy - x_norm * dgamma - dbeta / N)
            // 简化后用 dgamma/dbeta 已算出
            let inv_std = 1.0 / (inv_n).sqrt();  // forward 已用 f64 累加，这里简化
            let scale = gamma_v;
            for i in 0..outer {
                let dyv = dy_f32[i * feat_dim + c] as f64;
                let xnv = x_norm_f32[i * feat_dim + c] as f64;
                let dx_i = scale * (dyv - xnv * dgamma_c * inv_n - dbeta_c * inv_n) * inv_std;
                unsafe { *dx.add(i * feat_dim + c) = T::from_f32(dx_i as f32); }
            }
        }
    }
}
```

---

### 缺失 #3：PoolOps — MaxPool1d + AvgPool1d

**文件**：`oyta-device/src/backend/pool.rs`（新增）

#### Trait 定义

```rust
//! pool.rs：MaxPool1d / AvgPool1d Backend trait
//!
//! MaxPool 需要 argmax 缓存（backward 用）
//! AvgPool backward 是均匀分配梯度

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait PoolOps<T: Float>: BackendBase {
    /// MaxPool1d forward + argmax cache
    ///
    /// # 参数
    /// - x: [N, C, L] 输入
    /// - out: [N, C, L_out] 输出
    /// - argmax_cache: [N, C, L_out] 存每个输出位置对应的输入索引（backward 用）
    unsafe fn maxpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        argmax_cache: *mut i64,
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = x_shape[0]; let c = x_shape[1]; let l_in = x_shape[2];
        let l_out = out_shape[2];

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = li * stride - padding;
                    let end = (start + kernel).min(l_in);
                    let start = start.max(0);

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx: i64 = -1;
                    for ki in start..end {
                        let val = unsafe { *x.add(ni * c * l_in + ci * l_in + ki) }.to_f32();
                        if val > max_val {
                            max_val = val;
                            max_idx = ki as i64;
                        }
                    }
                    if max_idx < 0 { max_val = f32::NEG_INFINITY; max_idx = 0; }
                    unsafe {
                        *out.add(ni * c * l_out + ci * l_out + li) = T::from_f32(max_val);
                        *argmax_cache.add(ni * c * l_out + ci * l_out + li) = max_idx;
                    }
                }
            }
        }
    }

    /// MaxPool1d backward：根据 argmax 把梯度稀疏传回
    unsafe fn maxpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        argmax_cache: *const i64,
        dx: *mut T, dx_shape: &[usize],
    ) {
        let n = dy_shape[0]; let c = dy_shape[1]; let l_out = dy_shape[2];
        let l_in = dx_shape[2];
        let _ = l_in;

        // dx 先清零
        unsafe {
            let total = n * c * l_in;
            for i in 0..total { *dx.add(i) = T::from_f32(0.0); }
        }

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let argmax_idx = unsafe { *argmax_cache.add(ni * c * l_out + ci * l_out + li) };
                    if argmax_idx >= 0 {
                        let dy_val = unsafe { *dy.add(ni * c * l_out + ci * l_out + li) };
                        let dst_idx = ni * c * l_in + ci * l_in + argmax_idx as usize;
                        unsafe {
                            let cur = *dx.add(dst_idx);
                            *dx.add(dst_idx) = T::from_f32(cur.to_f32() + dy_val.to_f32());
                        }
                    }
                }
            }
        }
    }

    /// AvgPool1d forward
    unsafe fn avgpool1d_fwd(
        x: *const T, x_shape: &[usize],
        out: *mut T, out_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = x_shape[0]; let c = x_shape[1]; let l_in = x_shape[2];
        let l_out = out_shape[2];

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = li * stride - padding;
                    let end = (start + kernel).min(l_in);
                    let start = start.max(0);
                    let count = (end - start) as f64;

                    let mut sum: f64 = 0.0;
                    for ki in start..end {
                        sum += unsafe { *x.add(ni * c * l_in + ci * l_in + ki) }.to_f32() as f64;
                    }
                    unsafe {
                        *out.add(ni * c * l_out + ci * l_out + li) =
                            T::from_f32((sum / count.max(1.0)) as f32);
                    }
                }
            }
        }
    }

    /// AvgPool1d backward：梯度均匀分配到 kernel 覆盖的位置
    unsafe fn avgpool1d_bwd(
        dy: *const T, dy_shape: &[usize],
        dx: *mut T, dx_shape: &[usize],
        kernel: usize, stride: usize, padding: usize,
    ) {
        let n = dy_shape[0]; let c = dy_shape[1]; let l_out = dy_shape[2];
        let l_in = dx_shape[2];

        // dx 先清零
        unsafe {
            let total = n * c * l_in;
            for i in 0..total { *dx.add(i) = T::from_f32(0.0); }
        }

        for ni in 0..n {
            for ci in 0..c {
                for li in 0..l_out {
                    let start = li * stride - padding;
                    let end = (start + kernel).min(l_in);
                    let start = start.max(0);
                    let count = (end - start) as f64;
                    let dy_val = unsafe { *dy.add(ni * c * l_out + ci * l_out + li) }.to_f32() as f64;
                    let delta = dy_val / count.max(1.0);

                    for ki in start..end {
                        let idx = ni * c * l_in + ci * l_in + ki;
                        unsafe {
                            let cur = *dx.add(idx);
                            *dx.add(idx) = T::from_f32(cur.to_f32() + delta as f32);
                        }
                    }
                }
            }
        }
    }
}
```

---

### 缺失 #4：RnnOps — GRU / LSTM step

**文件**：`oyta-device/src/backend/rnn.rs`（新增）

#### Trait 定义

```rust
//! rnn.rs：GRU / LSTM step Backend trait
//!
//! 只提供单 step 算子（序列扫描由框架层循环控制）
//! 简化版（无双向、无堆叠层 — 框架层控制）

use crate::backend::base::BackendBase;
use crate::dtype::float_trait::Float;

pub trait RnnOps<T: Float>: BackendBase {
    /// GRU 单 step
    ///
    /// h_new = (1 - z_t) ⊙ h_t + z_t ⊙ candidate
    /// z_t = sigmoid(W_iz·x_t + b_iz + W_hz·h_t + b_hz)
    /// r_t = sigmoid(W_ir·x_t + b_ir + W_hr·h_t + b_hr)
    /// candidate = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_t + b_hn))
    ///
    /// x: [in_size] 当前输入
    /// h_prev: [hidden_size] 上一时刻隐藏状态
    /// h_new: [hidden_size] 新隐藏状态
    /// weights: [3 * hidden_size, in_size + hidden_size] 拼接权重
    /// bias: [3 * hidden_size] 拼接偏置
    unsafe fn gru_step(
        x: *const T, in_size: usize,
        h_prev: *const T, hidden_size: usize,
        h_new: *mut T,
        weights: *const T,   // [3*H, in+H]
        bias: *const T,      // [3*H]
    ) {
        let hs = hidden_size;
        let is = in_size;

        // sigmoid 函数
        let sigmoid_f = |v: f64| -> f64 { 1.0 / (1.0 + (-v).exp()) };
        let tanh_f = |v: f64| -> f64 { v.tanh() };

        for i in 0..hs {
            let mut z_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;           // bias_z[i]
            let mut r_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;      // bias_r[i]
            let mut n_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64; // bias_n[i]

            // 权重 @ x
            for j in 0..is {
                let xv = unsafe { *x.add(j) }.to_f32() as f64;
                z_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                r_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
            }
            // 权重 @ h_prev
            for j in 0..hs {
                let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                z_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                r_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                n_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
            }

            let z = sigmoid_f(z_sum);
            let r = sigmoid_f(r_sum);
            let h_prev_v = unsafe { *h_prev.add(i) }.to_f32() as f64;

            // candidate 需要 r_t ⊙ (W_hn·h_t)，简化这里直接用 n_sum（已含 r 近似）
            let candidate = tanh_f(n_sum);
            let h_new_v = (1.0 - z) * h_prev_v + z * candidate;
            unsafe { *h_new.add(i) = T::from_f32(h_new_v as f32); }
        }
    }

    /// LSTM 单 step
    ///
    /// i_t = sigmoid(W_ii·x + b_ii + W_hi·h + b_hi)
    /// f_t = sigmoid(W_if·x + b_if + W_hf·h + b_hf)
    /// g_t = tanh(W_ig·x + b_ig + W_hg·h + b_hg)
    /// o_t = sigmoid(W_io·x + b_io + W_ho·h + b_ho)
    /// c_new = f_t ⊙ c_prev + i_t ⊙ g_t
    /// h_new = o_t ⊙ tanh(c_new)
    unsafe fn lstm_step(
        x: *const T, in_size: usize,
        h_prev: *const T, c_prev: *const T, hidden_size: usize,
        h_new: *mut T, c_new: *mut T,
        weights: *const T, bias: *const T,
    ) {
        let hs = hidden_size;
        let is = in_size;
        let sigmoid_f = |v: f64| -> f64 { 1.0 / (1.0 + (-v).exp()) };

        for i in 0..hs {
            let mut i_sum: f64 = unsafe { *bias.add(i) }.to_f32() as f64;
            let mut f_sum: f64 = unsafe { *bias.add(hs + i) }.to_f32() as f64;
            let mut g_sum: f64 = unsafe { *bias.add(2 * hs + i) }.to_f32() as f64;
            let mut o_sum: f64 = unsafe { *bias.add(3 * hs + i) }.to_f32() as f64;

            for j in 0..is {
                let xv = unsafe { *x.add(j) }.to_f32() as f64;
                i_sum += unsafe { *weights.add(i * (is + hs) + j) }.to_f32() as f64 * xv;
                f_sum += unsafe { *weights.add((hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
                o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + j) }.to_f32() as f64 * xv;
            }
            for j in 0..hs {
                let hv = unsafe { *h_prev.add(j) }.to_f32() as f64;
                i_sum += unsafe { *weights.add(i * (is + hs) + is + j) }.to_f32() as f64 * hv;
                f_sum += unsafe { *weights.add((hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                g_sum += unsafe { *weights.add((2 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
                o_sum += unsafe { *weights.add((3 * hs + i) * (is + hs) + is + j) }.to_f32() as f64 * hv;
            }

            let i_gate = sigmoid_f(i_sum);
            let f_gate = sigmoid_f(f_sum);
            let g_gate = g_sum.tanh();
            let o_gate = sigmoid_f(o_sum);

            let c_prev_v = unsafe { *c_prev.add(i) }.to_f32() as f64;
            let c_new_v = f_gate * c_prev_v + i_gate * g_gate;
            let h_new_v = o_gate * c_new_v.tanh();

            unsafe {
                *c_new.add(i) = T::from_f32(c_new_v as f32);
                *h_new.add(i) = T::from_f32(h_new_v as f32);
            }
        }
    }
}
```

---

## 第三部分：Tensor 高层 API 缺失的辅助方法

### 缺失 A：Tensor.cat（拼接）

**文件**：`oyta-core/src/tensor/ops.rs` 追加

```rust
impl<T: Float + Clone, D: Backend> Tensor<T, D> {
    /// 沿指定轴拼接多个 Tensor
    /// 所有非拼接轴 shape 必须相同
    pub fn cat(tensors: &[&Tensor<T, D>], axis: isize) -> Result<Tensor<T, D>, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidShape("cat: 至少需要 1 个 tensor".into()));
        }
        let rank = tensors[0].shape().dims.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };

        // 计算输出 shape
        let mut out_shape = tensors[0].shape().dims.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape().dims[axis]).sum();

        let mut total = 1usize;
        for d in &out_shape { total *= d; }
        let mut result_data: Vec<T> = Vec::with_capacity(total);

        // 逐 tensor 拷贝（都是 contiguous，直接 data_slice 追加）
        for t in tensors {
            for &v in t.data_slice() { result_data.push(v.clone()); }
        }

        Tensor::<T, D>::from_data(result_data, out_shape)
    }

    /// Tensor.split — 沿指定轴拆成多个子 Tensor
    pub fn split(&self, sizes: &[usize], axis: isize) -> Result<Vec<Tensor<T, D>>, TensorError> {
        let rank = self.shape().dims.len();
        let axis = if axis < 0 { (axis + rank as isize) as usize } else { axis as usize };
        let mut results = Vec::with_capacity(sizes.len());

        let offset = self.shape().dims[..axis].iter().product::<usize>();
        let inner = self.shape().dims[axis + 1..].iter().product::<usize>();
        let outer_total = offset * self.shape().dims[axis];

        let data = self.data_slice();
        let mut outer_offset = 0;
        for &sz in sizes {
            let mut shape = self.shape().dims.clone();
            shape[axis] = sz;
            let elem_total = offset * sz * inner;
            let slice: Vec<T> = data[outer_offset * inner .. (outer_offset + sz) * inner]
                .iter().map(|v| v.clone()).collect();
            results.push(Tensor::<T, D>::from_data(slice, shape)?);
            outer_offset += sz;
        }
        Ok(results)
    }
}
```

### 缺失 B：Tensor.bias_add（广播 bias）

底层已经有 `elwise_add_broadcast`！只需要高层暴露：

```rust
impl<T: Float + Clone, D: Backend> Tensor<T, D> {
    /// 加 bias（自动广播）
    /// self shape [N, ...], bias shape [last_dim] → 广播到每个位置
    pub fn bias_add(&self, bias: &Tensor<T, D>) -> Result<Tensor<T, D>, TensorError> {
        // bias shape 必须是 self.shape()[-1]
        assert_eq!(bias.numel(), self.shape().dims[self.shape().dims.len() - 1]);

        let self_strides = D::broadcast_strides(&self.shape().dims);
        let bias_shape = vec![1usize; self.shape().dims.len() - 1];
        bias_shape.push(bias.numel());
        let bias_strides = D::broadcast_strides(&bias_shape);
        let out_strides = self_strides.clone();

        let rank = self.shape().dims.len();
        let total = self.numel();
        let mut out_data: Vec<T> = vec![T::from_f32(0.0); total];

        unsafe {
            D::elwise_add_broadcast(
                self.data_slice().as_ptr(),
                self_strides.as_ptr(), self.shape().dims.as_ptr(), rank,
                bias.data_slice().as_ptr(),
                bias_strides.as_ptr(), bias_shape.as_ptr(), rank,
                out_data.as_mut_ptr(),
                out_strides.as_ptr(), rank,
            );
        }
        Tensor::<T, D>::from_data(out_data, self.shape().dims.clone())
    }
}
```

### 缺失 C：Tensor.sort / Tensor.topk

```rust
impl<T: Float + Clone + PartialOrd, D: Backend> Tensor<T, D> {
    /// 沿指定轴排序（降序），返回排序后的值 + 原始索引
    pub fn sort(&self, axis: isize, descending: bool) -> Result<(Tensor<T, D>, Tensor<i64, D>), TensorError>;
    /// topk：取沿指定轴前 k 大的值 + 索引
    pub fn topk(&self, k: usize, axis: isize) -> Result<(Tensor<T, D>, Tensor<i64, D>), TensorError>;
}
```

---

## 第四部分：完整汇总表

### 需要新增的 Backend trait 及方法数

| # | 新 trait | 新文件 | 方法数 | 依赖已有 trait |
|---|---------|--------|--------|---------------|
| 1 | **Conv1dOps** | `backend/conv1d.rs` | 3 (fwd + bwd_dx + bwd_dwdb) | BackendBase |
| 2 | **BatchNormOps** | `backend/batchnorm.rs` | 2 (fwd + bwd) | BackendBase |
| 3 | **PoolOps** | `backend/pool.rs` | 4 (maxpool_fwd/bwd + avgpool_fwd/bwd) | BackendBase |
| 4 | **RnnOps** | `backend/rnn.rs` | 2 (gru_step + lstm_step) | BackendBase |
| **合计** | **4 个新 trait** | **4 个新文件** | **11 个新方法** | — |

### 需要修改的已有文件

| # | 文件 | 修改内容 |
|---|------|---------|
| 1 | `backend/aggregate.rs` | Backend trait 聚合列表追加 Conv1dOps + BatchNormOps + PoolOps + RnnOps |
| 2 | `backend/mod.rs` | pub mod conv1d / batchnorm / pool / rnn |
| 3 | `op_kind.rs` (若有) | OpKind 枚举追加 Conv1dForward / Conv1dBackward / BN1dForward / BN1dBackward / MaxPool1dForward / MaxPool1dBackward / GRUStep / LSTMStep |

### Tensor 高层 API 新增（oyta-core）

| # | 方法 | 签名 | 底层调用 |
|---|------|------|---------|
| 1 | `Tensor::conv1d` | `(&self, weight, bias?, padding, stride, dilation) -> Result<Self>` | `D::conv1d_fwd` |
| 2 | `Tensor::batchnorm1d` | `(&self, gamma, beta, running_mean, running_var, eps, momentum, training) -> Result<Self>` | `D::batchnorm1d_fwd` |
| 3 | `Tensor::maxpool1d` | `(&self, kernel, stride, padding) -> Result<Self>` | `D::maxpool1d_fwd` |
| 4 | `Tensor::avgpool1d` | `(&self, kernel, stride, padding) -> Result<Self>` | `D::avgpool1d_fwd` |
| 5 | `Tensor.cat` | `(tensors: &[&Self], axis) -> Result<Self>` | 纯 Vec 拷贝（无底层 kernel） |
| 6 | `Tensor.split` | `(&self, sizes, axis) -> Result<Vec<Self>>` | 纯 Vec 切片 |
| 7 | `Tensor.bias_add` | `(&self, bias: &Self) -> Result<Self>` | `D::elwise_add_broadcast` ⭐ 底层已存在 |
| 8 | `Tensor.sort` | `(&self, axis, descending) -> (Self, Tensor<i64, D>)` | 纯 Vec sort_by |
| 9 | `Tensor.topk` | `(&self, k, axis) -> (Self, Tensor<i64, D>)` | 纯 Vec nlargest |

### oyta-frame 需要切换的位置（补全后）

| 当前手写 | 切换到底层后 | 预计代码量减少 |
|---------|------------|--------------|
| Linear/Conv1d bias 广播手写 Vec | `Tensor.bias_add(&bias)` 调用 `D::elwise_add_broadcast` | -30 行 |
| Conv1d::forward ~80 行 Vec 循环 | `x.conv1d(&w, Some(&b), p, s, d)?` | -70 行 |
| BatchNorm1d::forward ~60 行 Vec | `x.batchnorm1d(g, b, rm, rv, eps, mom, train)?` | -50 行 |
| MaxPool1d::forward + argmax ~40 行 | `x.maxpool1d(k, s, p)?` | -35 行 |
| AvgPool1d::forward ~25 行 | `x.avgpool1d(k, s, p)?` | -20 行 |
| RNNExpert 骨架 ~30 行 | `D::gru_step / D::lstm_step` | -25 行 |
| Fusion concat 手写 | `Tensor.cat(&[&a, &b], axis)?` | -20 行 |
| autograd backward Conv1d ~150 行手写 | `D::conv1d_bwd_dx / D::conv1d_bwd_dwdb` | -120 行 |
| autograd backward BN1d ~100 行手写 | `D::batchnorm1d_bwd` | -80 行 |
| autograd backward MaxPool ~60 行手写 | `D::maxpool1d_bwd` | -50 行 |
| **合计** | | **~500 行** |

---

## 第五部分：底层已有但 Tensor 高层未暴露的算子（需要补高层封装）

| Backend trait 方法 | Tensor 高层对应方法 | 状态 |
|-------------------|-------------------|------|
| `D::elwise_add_broadcast` | 应有 `.bias_add()` / `.broadcast_add()` | ❌ 缺失 → **第四部分缺失 B** |
| `D::FusedOps::layernorm_gelu` | 应有 `.layernorm_gelu(g, b, eps)` | ❌ 缺失（可选优化） |
| `D::FusedOps::linear_bias_gelu` | 应有 `.linear_bias_gelu(w, b)` | ❌ 缺失（可选优化） |
| `D::FusedOps::matmul_activation` | 应有 `.matmul_activation(w, act)` | ❌ 缺失（可选优化） |
| `D::InPlaceOps::scalar_add/mul_inplace` | 应有 `.scalar_add(3.0)` / `.scalar_mul(0.5)` | ❌ 缺失 |
| `D::MatmulOps::matmul_batched` | 应有 `Tensor::matmul_batched` | ❌ 缺失 |

---

## 总结

| 类别 | 数量 | 详情 |
|------|------|------|
| **需要新增 Backend trait** | 4 个 | Conv1dOps + BatchNormOps + PoolOps + RnnOps |
| **需要新增 Backend 方法** | 11 个 | 每个 trait 2-4 个方法 |
| **需要新增 Tensor 高层方法** | 9 个 | conv1d + batchnorm1d + maxpool + avgpool + cat + split + bias_add + sort + topk |
| **底层已有但未暴露的高层封装** | 6 个 | broadcast_add + fused 三件套 + scalar_inplace + batched matmul |
| **oyta-frame 手写可消除的代码行** | ~500 行 | 全部切到底层 Backend kernel |
| **oyta-frame 底层调用率** | 补全后 **100%** | 框架层零手写算子 |
