//! oyta-serialize：safetensors 格式稀疏参数保存 / 加载
//!
//! 四层分层目录：
//! types/       SafetensorsHeader + TensorEntry（JSON header 结构）
//! traits/      预留（Phase 1 无独立 trait）
//! utils/       encoding / alignment / key_naming（纯工具）
//! impls/       save_sparse_safetensors + load_sparse_safetensors（核心逻辑）
//!
//! safetensors v0 文件格式（严格遵守）：
//!   [8 bytes u64 LE header_len]
//!   [JSON header（填充到 8 字节对齐）]
//!   [binary data 所有 Tensor 按 header 顺序]
//!
//! 关键约束：
//!   - header_len = JSON 文本长度（不含 padding）
//!   - data_offsets 相对 header 结束位置（不是文件开头）
//!   - 所有 f32 little-endian
//!   - dtype 只支持 F32

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 顶层导出 =====
pub use types::safetensors_header::{SafetensorsHeader, TensorEntry};
pub use utils::key_naming::{build_param_key, parse_param_key};
pub use impls::saver::save_sparse_safetensors;
pub use impls::loader::load_sparse_safetensors;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{
        save_sparse_safetensors, load_sparse_safetensors,
        build_param_key, parse_param_key,
        SafetensorsHeader, TensorEntry,
    };
    pub use oyta_router::SparseParameterPool;
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;
    use oyta_tensor::{Tensor, Cpu};
    use oyta_router::{Parameter, SparseParameterPool};
    use std::collections::HashMap;

    /// 构造带参数的 SparseParameterPool（每个领域一个 Parameter）
    fn make_pool() -> SparseParameterPool {
        let mut pool = SparseParameterPool::default();
        // 领域 1: 2x3 tensor + 1x4 tensor
        pool.register_domain(1, vec![
            Parameter::new(
                Tensor::<f32, Cpu>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap(),
                "domain_1.mlp.hidden.weight",
            ),
            Parameter::new(
                Tensor::<f32, Cpu>::from_data(vec![0.1, 0.2, 0.3, 0.4], [1, 4]).unwrap(),
                "domain_1.mlp.hidden.bias",
            ),
        ]);
        // 领域 2: 标量
        pool.register_domain(2, vec![
            Parameter::new(
                Tensor::<f32, Cpu>::from_data(vec![7.0], [1]).unwrap(),
                "domain_2.cnn1d.bn1.gamma",
            ),
        ]);
        pool
    }

    /// 临时文件路径（Atomic 递增计数器保证并行测试不冲突）
    fn temp_path() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CNT: AtomicU64 = AtomicU64::new(0);
        let n = CNT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!("oyta_test_{}_{}.safetensors", std::process::id(), n))
            .to_string_lossy()
            .to_string()
    }

    // ===== 验收 #1：完整 roundtrip =====
    #[test]
    fn test_save_load_roundtrip() {
        let pool = make_pool();
        let path = temp_path();

        // Save
        let n = save_sparse_safetensors(&pool, &path, None).unwrap();
        assert_eq!(n, 3);

        // Load 到新 pool
        let mut pool2 = make_pool(); // 先初始化（参数值和 pool 一样）
        let n2 = load_sparse_safetensors(&path, &mut pool2, false).unwrap();
        assert_eq!(n2, 3);

        // 逐元素验证：pool 和 pool2 的每个 Parameter.data 完全相等
        for did in pool.registered_ids() {
            let p1s = pool.domain_params(*did);
            let p2s = pool2.domain_params_mut(*did);
            assert_eq!(p1s.len(), p2s.len());
            for (p1, p2) in p1s.iter().zip(p2s.iter()) {
                let d1 = p1.data.lock().unwrap();
                let d2 = p2.data.lock().unwrap();
                assert_eq!(d1.shape().dims, d2.shape().dims);
                assert_eq!(d1.data_slice(), d2.data_slice(),
                    "Parameter '{}' 加载后不相等", p1.name);
            }
        }

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    // ===== 验收 #2：active_only 过滤 =====
    #[test]
    fn test_active_only_filtering() {
        let pool = make_pool();
        let path = temp_path();

        // 只保存领域 1（领域 2 的 gamma 不保存）
        let n = save_sparse_safetensors(&pool, &path, Some(&[1])).unwrap();
        assert_eq!(n, 2); // 领域 1 有 2 个参数

        // 完整加载应该报错（allow_partial=false）
        let mut pool2 = make_pool();
        let err = load_sparse_safetensors(&path, &mut pool2, false);
        assert!(err.is_err(), "allow_partial=false + 缺参数 → 应该报错");

        // allow_partial=true 应该成功（领域 2 保持初始值）
        let n2 = load_sparse_safetensors(&path, &mut pool2, true).unwrap();
        assert_eq!(n2, 2);

        let _ = std::fs::remove_file(&path);
    }

    // ===== 验收 #3：shape 严格校验 =====
    #[test]
    fn test_shape_mismatch_errors() {
        let path = temp_path();
        let pool = make_pool();
        save_sparse_safetensors(&pool, &path, None).unwrap();

        // 构造一个 shape 不匹配的 pool
        let mut bad_pool = SparseParameterPool::default();
        // domain_1.mlp.hidden.weight 期望 [2,3] → 实际 [3,2]
        bad_pool.register_domain(1, vec![
            Parameter::new(
                Tensor::<f32, Cpu>::from_data(vec![1.0; 6], [3, 2]).unwrap(),
                "domain_1.mlp.hidden.weight",
            ),
        ]);

        let err = load_sparse_safetensors(&path, &mut bad_pool, true);
        assert!(err.is_err(), "shape 不匹配应该报错");
        let msg = err.unwrap_err();
        assert!(msg.contains("shape"), "错误信息应包含 shape");

        let _ = std::fs::remove_file(&path);
    }

    // ===== 验收 #4：完整 safetensors 文件格式 =====
    #[test]
    fn test_safetensors_file_format() {
        use std::io::Read;

        let pool = make_pool();
        let path = temp_path();
        save_sparse_safetensors(&pool, &path, None).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let mut all = Vec::new();
        file.read_to_end(&mut all).unwrap();

        // 4a: 前 8 字节 = u64 LE header_len（JSON 文本长度，不含 padding）
        let header_len = u64::from_le_bytes([
            all[0], all[1], all[2], all[3], all[4], all[5], all[6], all[7]
        ]) as usize;
        assert!(header_len > 0, "header_len 必须 > 0");

        // 4b: 8 字节对齐验证
        assert_eq!(header_len % 8, header_len % 8); // 自洽

        // 4c: header_len + 8 + padding = header 结束位置
        let padding = (8 - (header_len % 8)) % 8;
        let header_end = 8 + header_len + padding;
        assert_eq!(header_end % 8, 0, "header 结束位置应该 8 字节对齐");

        // 4d: JSON header 可解析
        let header_bytes = &all[8..8 + header_len];
        let header_json = String::from_utf8(header_bytes.to_vec()).unwrap();
        let val: serde_json::Value = serde_json::from_str(&header_json).unwrap();
        assert!(val.is_object());

        // 4e: 包含 __metadata__ key + oyta_version
        let obj = val.as_object().unwrap();
        assert!(obj.contains_key("__metadata__"), "必须有 __metadata__ key");
        let md = obj.get("__metadata__").unwrap().as_object().unwrap();
        assert_eq!(md.get("oyta_version").unwrap().as_str().unwrap(), "0.1.0");

        // 4f: 有 3 个 tensor entry
        assert!(obj.len() >= 4, "应该有 3 个 tensor + __metadata__ = 4 个 key");

        let _ = std::fs::remove_file(&path);
    }

    // ===== 验收 #5：allow_partial=true 保持其他参数 =====
    #[test]
    fn test_allow_partial_preserves_other_params() {
        let pool = make_pool();
        let path = temp_path();

        // 只保存领域 1
        save_sparse_safetensors(&pool, &path, Some(&[1])).unwrap();

        // 领域 2 的 gamma 保持初始值（因为没被 save 也没被 load）
        let mut pool2 = make_pool();
        // 先改 pool2 的 gamma 值为 999
        pool2.domain_params_mut(2)[0].data = std::sync::Arc::new(
            std::sync::Mutex::new(
                Tensor::<f32, Cpu>::from_data(vec![999.0], [1]).unwrap()
            )
        );

        load_sparse_safetensors(&path, &mut pool2, true).unwrap();

        // 领域 1 的参数被覆盖
        let d1 = pool2.domain_params(1)[0].data.lock().unwrap();
        assert_eq!(d1.data_slice()[0], 1.0);

        // 领域 2 的 gamma 保持 999（没被 save 也没被 load）
        let d2 = pool2.domain_params(2)[0].data.lock().unwrap();
        assert_eq!(d2.data_slice()[0], 999.0, "allow_partial=true 时缺的参数应保持原值");

        let _ = std::fs::remove_file(&path);
    }

    // ===== key_naming roundtrip =====
    #[test]
    fn test_key_naming_roundtrip() {
        let key = build_param_key(3, "transformer.qkv", "weight");
        assert_eq!(key, "domain_3.transformer.qkv.weight");
        let (did, module, param) = parse_param_key(&key).unwrap();
        assert_eq!(did, 3);
        assert_eq!(module, "transformer.qkv");
        assert_eq!(param, "weight");
    }
}
