//! concat_tensors：任意 dim 拼接 Tensor
//!
//! 纯 Vec 实现（oyta-tensor 0.1.1 Backend trait 未暴露 concat 高层 API）
//! 支持 2D/3D/任意 rank Tensor 的 dim=0/1/2/... 拼接

use oyta_tensor::prelude::*;

/// 在指定 dim 上拼接多个 Tensor
///
/// 约束：所有输入 Tensor 必须 rank 相同、非拼接 dim 上 shape 相同
/// 例：[batch, d1] concat [batch, d2] at dim=1 → [batch, d1+d2]
pub fn concat_tensors(tensors: &[&Tensor<f32, Cpu>], dim: usize) -> Tensor<f32, Cpu> {
    assert!(!tensors.is_empty(), "concat_tensors 输入不能为空");
    let rank = tensors[0].rank();
    assert!(dim < rank, "concat_tensors dim={dim} 超过 rank={rank}");

    // Step 1: 验证 shape 兼容性 + 计算输出 shape
    let mut out_shape = tensors[0].shape().dims.clone();
    for t in tensors.iter().skip(1) {
        assert_eq!(t.rank(), rank, "concat_tensors rank 必须一致");
        let tdim = t.shape().dims.clone();
        for i in 0..rank {
            if i != dim {
                assert_eq!(tdim[i], out_shape[i],
                    "concat_tensors 非拼接 dim={i} shape 不匹配: {} vs {}", tdim[i], out_shape[i]);
            } else {
                out_shape[dim] += tdim[dim];
            }
        }
    }

    // Step 2: 按行 flatten 所有输入，按 dim 位置交错拷贝
    let total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; total];
    let out_strides = compute_strides(&out_shape);

    // 逐 Tensor、逐元素拷贝到输出正确位置
    let mut dim_offset = 0usize; // 当前 dim 已消费的偏移
    for t in tensors {
        let td = t.data_slice();
        let tshape = t.shape().dims.clone();
        let tstrides = compute_strides(&tshape);
        let trank = tshape.len();

        // 遍历输出的所有位置
        let num_out = total;
        for out_idx in 0..num_out {
            // 解包 out_idx → 各 dim 坐标
            let mut out_coords = vec![0usize; trank];
            let mut rem = out_idx;
            for d in 0..trank {
                out_coords[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            // 如果当前 dim 坐标落在这个 Tensor 的范围内 → 拷贝
            let local_dim = out_coords[dim] as isize - dim_offset as isize;
            if local_dim >= 0 && (local_dim as usize) < tshape[dim] {
                out_coords[dim] = local_dim as usize; // 转换为 Tensor 内的局部坐标
                // 计算 flat index in t
                let mut flat_idx = 0usize;
                for d in 0..trank {
                    flat_idx += out_coords[d] * tstrides[d];
                }
                out[out_idx] = td[flat_idx];
            }
        }
        dim_offset += tshape[dim];
    }

    Tensor::<f32, Cpu>::from_data(out, out_shape)
        .expect("concat_tensors 输出构造失败")
}

/// 计算各维 stride（从左到右）
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
