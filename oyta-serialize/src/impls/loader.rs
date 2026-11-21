//! load_sparse_safetensors：按 key 匹配加载（严格 shape 校验 + 部分加载支持）
//!
//! 核心特性：
//!   - 严格 shape 校验（不匹配返回 error）
//!   - 支持部分加载（文件只有部分领域参数 → 其他保持初始化值）
//!   - 严格字节长度校验（防止 header data_offsets 与实际 bytes 不匹配）
//!   - key = Parameter.name（build_param 统一约定：domain_{did}.{module}.{param}）

use std::fs::File;
use std::io::Read;
use oyta_router::SparseParameterPool;
use oyta_tensor::prelude::*;
use crate::types::safetensors_header::SafetensorsHeader;
use crate::utils::encoding::bytes_to_f32_vec;

/// 加载 safetensors 文件到 SparseParameterPool
///
/// # 参数
/// - `path`: 文件路径
/// - `pool`: 参数池（按 Parameter.name 匹配覆盖 data）
/// - `allow_partial`: true = 文件只有部分参数也能加载（缺的保持初始化值）；
///                    false = 文件缺 pool 里任何一个参数都报错
///
/// # 返回
/// 成功返回加载的 tensor 数量；失败返回 error 字符串
pub fn load_sparse_safetensors(
    path: &str,
    pool: &mut SparseParameterPool,
    allow_partial: bool,
) -> Result<usize, String> {
    // ===== Step 1: 打开文件 + 读 header_len =====
    let mut file = File::open(path).map_err(|e| format!("打开文件失败: {}", e))?;

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("读 header_len 失败: {}", e))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    // ===== Step 2: 读 JSON header（header_len 字节）+ 跳过 8 字节对齐 padding =====
    let mut header_json_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_json_bytes)
        .map_err(|e| format!("读 header JSON 失败: {}", e))?;
    let header_json = String::from_utf8(header_json_bytes)
        .map_err(|e| format!("header 不是合法 UTF-8: {}", e))?;

    // 跳过 8 字节对齐 padding
    let padding = (8 - (header_len % 8)) % 8;
    if padding > 0 {
        let mut skip_buf = vec![0u8; padding];
        file.read_exact(&mut skip_buf)
            .map_err(|e| format!("跳过 header padding 失败: {}", e))?;
    }

    // ===== Step 3: 解析 SafetensorsHeader =====
    let header_val: serde_json::Value = serde_json::from_str(&header_json)
        .map_err(|e| format!("解析 header JSON 失败: {}", e))?;
    let header = SafetensorsHeader::from_json(&header_val)?;

    // ===== Step 4: 读全部 binary data =====
    let mut all_data = Vec::new();
    file.read_to_end(&mut all_data)
        .map_err(|e| format!("读 binary data 失败: {}", e))?;

    // ===== Step 5: 遍历 pool Parameter，按 name 匹配 + 覆盖 data =====
    // 先 collect registered_ids 避免 immutable + mutable borrow 冲突
    let dids: Vec<u32> = pool.registered_ids().iter().copied().collect();
    let mut loaded: usize = 0;
    let mut skipped: usize = 0;
    let mut shape_errors: Vec<String> = Vec::new();

    for did in dids {
        for p in pool.domain_params_mut(did) {
            if let Some(entry) = header.tensors.get(&p.name) {
                // Shape 严格校验
                let current_shape = p.data.lock().unwrap().shape().dims.clone();
                if entry.shape != current_shape {
                    shape_errors.push(format!(
                        "'{}': 文件 shape {:?} ≠ 当前 shape {:?}",
                        p.name, entry.shape, current_shape
                    ));
                    continue;
                }

                let [start, end] = entry.data_offsets;
                // 边界检查
                if start > end || end > all_data.len() {
                    return Err(format!(
                        "'{}': data_offsets [{}, {}] 越界 (binary len={})",
                        p.name, start, end, all_data.len()
                    ));
                }

                let bytes = &all_data[start..end];
                let numel: usize = entry.shape.iter().product();
                // 字节长度严格校验
                if bytes.len() != numel * 4 {
                    return Err(format!(
                        "'{}': 字节长度不匹配: 期望 {}, 实际 {}",
                        p.name, numel * 4, bytes.len()
                    ));
                }

                // bytes → f32 Vec → Tensor → 覆盖 Parameter.data
                let f32_data = bytes_to_f32_vec(bytes)?;
                let new_tensor = Tensor::<f32, Cpu>::from_data(f32_data, entry.shape.clone())
                    .map_err(|e| format!("构造 Tensor 失败 '{}': {}", p.name, e))?;
                *p.data.lock().unwrap() = new_tensor;
                loaded += 1;
            } else {
                // 文件里没有这个参数
                if !allow_partial {
                    return Err(format!(
                        "文件缺参数 '{}' (partial loading not allowed)",
                        p.name
                    ));
                }
                skipped += 1;
            }
        }
    }

    // 有 shape 不匹配 → 返回 error（即使 allow_partial）
    if !shape_errors.is_empty() {
        return Err(format!("shape 不匹配: {}", shape_errors.join("; ")));
    }

    println!(
        "[oyta-serialize] 加载 {} 个 tensor (skipped={}, allow_partial={})",
        loaded, skipped, allow_partial
    );
    Ok(loaded)
}
