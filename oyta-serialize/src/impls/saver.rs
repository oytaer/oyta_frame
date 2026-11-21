//! save_sparse_safetensors：safetensors 完整二进制格式实现
//!
//! 文件格式（严格对齐 safetensors v0 规范）：
//!   [8 bytes u64 LE header_len]  ← JSON 文本长度（不含 8 字节对齐 padding）
//!   [JSON header（填充到 8 字节对齐）]
//!   [binary data 所有 Tensor 按 header 顺序]
//!
//! key 命名（oyta_router::param::Parameter.name 已统一）：
//!   domain_{did}.{module}.{param}
//!   例：domain_1.mlp.hidden.weight

use std::fs::File;
use std::io::Write;
use oyta_router::SparseParameterPool;
use crate::types::safetensors_header::{SafetensorsHeader, TensorEntry};
use crate::utils::encoding::f32_slice_to_bytes;
use crate::utils::alignment::pad_to_align;

/// 保存参数到 safetensors 文件
///
/// # 参数
/// - `pool`: 参数池（所有领域的全部参数）
/// - `path`: 输出文件路径
/// - `active_only`: None = 全部领域；Some = 只保存指定领域（稀疏训练省磁盘）
pub fn save_sparse_safetensors(
    pool: &SparseParameterPool,
    path: &str,
    active_only: Option<&[u32]>,
) -> Result<usize, String> {
    // ===== Step 1: 收集所有 (key, bytes, shape) =====
    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    let domain_filter: Option<Vec<u32>> = active_only.map(|v| v.to_vec());

    for did in pool.registered_ids() {
        // active_only 过滤（稀疏训练省磁盘）
        if let Some(ref f) = domain_filter { if !f.contains(did) { continue; } }

        for p in pool.domain_params(*did) {
            let data = p.data.lock().unwrap();
            let shape = data.shape().dims.clone();
            let d = data.data_slice();
            let bytes = f32_slice_to_bytes(d); // f32 → LE bytes
            tensors.push((p.name.clone(), bytes, shape));
        }
    }

    if tensors.is_empty() {
        println!("[oyta-serialize] 警告：无参数可保存");
        return Ok(0);
    }

    // ===== Step 2: 组 SafetensorsHeader（JSON header）=====
    let mut header = SafetensorsHeader::new();
    let mut current: usize = 0; // 当前 binary 偏移（相对 header 结束位置）
    for (key, bytes, shape) in &tensors {
        let len = bytes.len();
        let entry = TensorEntry {
            dtype: "F32".to_string(),
            shape: shape.clone(),
            data_offsets: [current, current + len],
        };
        header.tensors.insert(key.clone(), entry);
        current += len;
    }

    // ===== Step 3: 序列化 header JSON + 8 字节对齐 =====
    let header_json = serde_json::to_string(&header.to_json())
        .map_err(|e| format!("序列化 header 失败: {}", e))?;
    let header_len = header_json.len() as u64; // ⚠️ JSON 文本长度（不含 padding）
    let mut header_bytes = header_json.into_bytes();
    pad_to_align(&mut header_bytes); // 填充到 8 字节对齐

    // ===== Step 4: 写文件 =====
    let mut file = File::create(path).map_err(|e| format!("创建文件失败: {}", e))?;

    // 4a: 8 bytes u64 LE header_len（JSON 长度，不含 padding）
    file.write_all(&header_len.to_le_bytes())
        .map_err(|e| format!("写 header_len 失败: {}", e))?;

    // 4b: 填充后的 header bytes
    file.write_all(&header_bytes)
        .map_err(|e| format!("写 header 失败: {}", e))?;

    // 4c: 所有 tensor 的 binary data（按收集顺序）
    for (_, bytes, _) in &tensors {
        file.write_all(bytes)
            .map_err(|e| format!("写 tensor data 失败: {}", e))?;
    }
    file.flush().map_err(|e| format!("flush 失败: {}", e))?;

    println!(
        "[oyta-serialize] 保存 {} 个 tensor 到 {} (header_len={}, total_bytes={}, oyta_version=0.1.0)",
        tensors.len(), path, header_len, current
    );
    Ok(tensors.len())
}
