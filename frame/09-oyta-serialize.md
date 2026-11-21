# oyta-serialize 开发文档（Phase 9）

> crate：`oyta-serialize/` | 依赖：**`oyta-tensor = "0.1.1"` + `oyta-router` + `oyta-expert` + `safetensors = "0.6.0"`**
> 职责：safetensors 格式按领域 key 分层存储 + 部分加载

---

# 一、Cargo.toml

```toml
[package]
name = "oyta-serialize"
version = "0.1.0"
edition = "2024"

[dependencies]
oyta-tensor = "0.1.1"
oyta-router = { version = "0.1.0", path = "../oyta-router" }
oyta-expert = { version = "0.1.0", path = "../oyta-expert" }
safetensors = "0.6.0"   # crates.io 最新
serde       = { version = "1.0.229", features = ["derive"] }
serde_json  = "1.0.152"
thiserror   = "2.0.16"
anyhow      = "1.0.93"
```

---

# 二、src/lib.rs

```rust
pub mod saver;
pub mod loader;

pub use saver::save_sparse_safetensors;
pub use loader::load_sparse_safetensors;

pub mod prelude {
    pub use crate::{save_sparse_safetensors, load_sparse_safetensors};
    pub use oyta_router::SparseParameterPool;
}
```

---

# 三、src/saver.rs

```rust
//! save_sparse_safetensors：safetensors 完整二进制格式实现
//!
//! 文件格式：
//!   [8 bytes u64 LE header_len]
//!   [JSON header（填充到 8 字节对齐）]
//!   [binary data 全部 Tensor 按 header 顺序]
//!
//! key 命名规则（oyta_router::param::build_param 已统一）：
//!   domain_{did}.{module}.{param}
//!   例：domain_1.mlp.hidden.weight

use std::fs::File;
use std::io::Write;
use oyta_router::SparseParameterPool;

/// 保存参数到 safetensors 文件
///
/// # 参数
/// - pool: 参数池（所有领域的全部参数）
/// - path: 输出路径
/// - active_only: None = 全部领域；Some = 只保存指定领域（稀疏训练省磁盘）
pub fn save_sparse_safetensors(
    pool: &SparseParameterPool,
    path: &str,
    active_only: Option<&[u32]>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: 收集 (key, bytes, shape)
    let domain_filter: Option<Vec<u32>> = active_only.map(|v| v.to_vec());
    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for did in pool.registered_ids() {
        if let Some(ref f) = domain_filter { if !f.contains(did) { continue; } }
        for p in pool.domain_params(*did) {
            let data = p.data.lock().unwrap();
            let shape = data.shape().dims.clone();
            let numel = data.numel();
            let d = data.data_slice();
            let mut bytes = Vec::with_capacity(numel * 4);
            for &v in d { bytes.extend_from_slice(&v.to_le_bytes()); }
            tensors.push((p.name.clone(), bytes, shape));
        }
    }

    if tensors.is_empty() {
        println!("[oyta-serialize] 警告：无参数可保存");
        return Ok(());
    }

    // Step 2: 组 safetensors header JSON
    let mut header_map: serde_json::Value = serde_json::json!({});
    let mut current: usize = 0;
    for (key, bytes, shape) in &tensors {
        let len = bytes.len();
        header_map[key] = serde_json::json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [current, current + len],
        });
        current += len;
    }

    // Step 3: 序列化 + 8 字节对齐 header
    let header_json = serde_json::to_string(&header_map)?;
    let header_len = header_json.len() as u64;
    let padding = (8 - (header_len % 8)) % 8;
    let mut header_bytes = header_json.into_bytes();
    header_bytes.extend(std::iter::repeat(b' ').take(padding as usize));

    // Step 4: 写文件
    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;  // 8 bytes header_len
    file.write_all(&header_bytes)?;               // 填充后的 header
    for (_, bytes, _) in &tensors { file.write_all(bytes)?; }  // binary data
    file.flush()?;

    println!("[oyta-serialize] saved {} tensors to {}", tensors.len(), path);
    Ok(())
}
```

---

# 四、src/loader.rs

```rust
//! load_sparse_safetensors：按 key 匹配加载
//!
//! 核心特性：
//!   - 严格 shape 校验（不匹配返回 error）
//!   - 支持部分加载（文件只有部分领域参数 → 其他保持初始化值）
//!   - key = param.name（build_param 统一约定）

use std::fs::File;
use std::io::Read;
use oyta_router::SparseParameterPool;
use oyta_tensor::prelude::*;

pub fn load_sparse_safetensors(
    path: &str,
    pool: &mut SparseParameterPool,
    allow_partial: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Step 1: 读 header
    let mut file = File::open(path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header_json = String::from_utf8(header_bytes)?;
    let header: serde_json::Value = serde_json::from_str(&header_json)?;

    // Step 2: 收集 file 里的 Tensor 信息
    let mut file_tensors: std::collections::HashMap<String, (Vec<usize>, (usize, usize))> = std::collections::HashMap::new();
    for (key, val) in header.as_object().unwrap() {
        let shape: Vec<usize> = val.get("shape").and_then(|s| s.as_array())
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let offsets = val.get("data_offsets").and_then(|o| o.as_array())
            .map(|a| (a[0].as_u64().unwrap_or(0) as usize, a[1].as_u64().unwrap_or(0) as usize))
            .unwrap_or((0, 0));
        file_tensors.insert(key.clone(), (shape, offsets));
    }

    // Step 3: 读全部 binary
    let mut all_data = Vec::new();
    file.read_to_end(&mut all_data)?;

    // Step 4: 遍历 pool 里的 Parameter，按 name 匹配 + 覆盖 data
    let mut loaded: usize = 0;
    let mut skipped: usize = 0;

    for did in pool.registered_ids().iter().copied() {
        for p in pool.domain_params_mut(did) {
            if let Some((target_shape, (start, end))) = file_tensors.get(&p.name) {
                // shape 严格校验
                let current_shape = p.data.lock().unwrap().shape().dims.clone();
                if target_shape != &current_shape {
                    return Err(format!(
                        "oyta-serialize: shape mismatch '{}': file {:?}, current {:?}",
                        p.name, target_shape, current_shape
                    ).into());
                }

                let bytes = &all_data[*start..*end];
                let numel = target_shape.iter().product::<usize>();
                if bytes.len() != numel * 4 {
                    return Err(format!(
                        "oyta-serialize: byte size mismatch '{}': expected {}, got {}",
                        p.name, numel * 4, bytes.len()
                    ).into());
                }

                // bytes → f32 Vec → Tensor → 覆盖 Parameter.data
                let mut f32_data = vec![0.0f32; numel];
                for i in 0..numel {
                    let arr: [u8; 4] = [bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3]];
                    f32_data[i] = f32::from_le_bytes(arr);
                }
                let new_tensor = Tensor::<f32, Cpu>::from_data(f32_data, target_shape.iter().copied()).unwrap();
                *p.data.lock().unwrap() = new_tensor;
                loaded += 1;
            } else {
                if !allow_partial {
                    return Err(format!(
                        "oyta-serialize: param '{}' not found (partial loading not allowed)",
                        p.name
                    ).into());
                }
                skipped += 1;
            }
        }
    }

    println!("[oyta-serialize] loaded {} tensors (skipped {}, allow_partial={})", loaded, skipped, allow_partial);
    Ok(loaded)
}
```

---

# 五、oyta-serialize 验收标准

```
[ ] cargo check --package oyta-serialize 零警告
[ ] Cargo.toml 只有 oyta-tensor = "0.1.1" + safetensors + workspace path（无 rust-version）
[ ] save_sparse_safetensors 保存后文件结构正确：
    8 bytes u64 LE header_len + JSON header（8 字节对齐）+ binary data
[ ] load_sparse_safetensors 加载后：
    → Parameter.data 逐元素和保存前相等
    → shape 严格校验（不匹配返回 error）
[ ] allow_partial = false + 文件缺某个参数 → error
[ ] allow_partial = true + 文件只含部分领域 → 加载成功，缺的参数保持初始化值
[ ] active_only = Some(&[1, 2]) → 只保存领域 1 和 2，省磁盘
[ ] 每个函数有中文注释
```

---

# 六、四层分层目录规范（对齐 00-overview §八）

oyta-serialize 当前是平铺 `src/` 结构，需改为 types/traits/utils/impl 四层分层。核心：safetensors header 结构 + key 命名归入 types，二进制编码/解码纯工具归入 utils，save/load 核心逻辑归入 impl。

## 6.1 目标目录结构

```
oyta-serialize/src/
├── lib.rs                       ← 顶层统一 pub mod + pub use + prelude
│
├── types/                       ← 数据类型层（纯结构体）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── safetensors_header.rs   ← SafetensorsHeader { tensors: Vec<TensorEntry> }、TensorEntry { name, dtype, shape, offsets }
│
├── traits/                      ← 接口层（预留扩展）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   └── (预留：如果后续有 Serializer trait 抽象)
│
├── utils/                       ← 工具层（纯二进制编解码）
│   ├── mod.rs                  ← 只 pub mod xxx;
│   ├── encoding.rs             ← bytes ↔ f32 little-endian 编解码
│   ├── alignment.rs            ← 8 字节对齐填充（safetensors header 要求）
│   └── key_naming.rs           ← domain_{did}.{module}.{param} key 生成 / 解析
│
└── impl/                        ← 实现层（核心 save/load）
    ├── mod.rs                  ← 只 pub mod xxx;
    ├── saver.rs                ← save_sparse_safetensors：遍历 SparseParameterPool + 组 header + 写二进制文件
    └── loader.rs               ← load_sparse_safetensors：读文件 → 解析 header → 按 key 匹配 → 覆盖 Parameter.data
```

## 6.2 safetensors 文件格式规范（必须严格遵守）

```
完整文件格式（safetensors v0 规范）：

┌─────────────────────────────────────────────┐
│ [8 bytes u64 LE]  header_len (JSON 长度)     │
│ [JSON header]  填充到 8 字节对齐             │
│   {                                         │
│     "domain_1.mlp.hidden.weight": {         │
│       "dtype": "F32",                       │
│       "shape": [128, 64],                   │
│       "data_offsets": [0, 32768]            │
│     },                                      │
│     "domain_1.mlp.hidden.bias": { ... },    │
│     "__metadata__": { "oyta_version": "0.1.0" } │
│   }                                         │
│ [binary data]  按 header 顺序依次拼接         │
└─────────────────────────────────────────────┘

⚠️ 关键约束：
- header_len 是 JSON 文本长度（不含填充），不是 8 字节对齐后的长度
- header 结束位置 = 8 + header_len + padding（padding = (8 - header_len % 8) % 8）
- data_offsets 是相对 header 结束位置的偏移（不是相对文件开头）
- 所有数值用 little-endian（u32 for f32 bytes, u64 for offsets）
- dtype 目前只支持 F32（oyta-frame 统一 f32 精度）
```

## 6.3 key 命名规范（统一约定）

```rust
//! utils/key_naming.rs
//! Parameter.name = "domain_{did}.{module}.{param}"
//! 例：domain_1.mlp.hidden.weight / domain_2.cnn1d.bn1.gamma / domain_0.trans.qkv.weight
//!
//! SparseParameterPool.register_domain(did, params) 时，每个 Parameter 的 name
//! 必须遵循此格式，否则 save/load 无法正确按 key 匹配

pub fn build_param_key(did: u32, module: &str, param: &str) -> String {
    format!("domain_{}.{}.{}", did, module, param)
}

pub fn parse_param_key(key: &str) -> Option<(u32, &str, &str)> {
    // "domain_1.mlp.hidden.weight" → (1, "mlp.hidden", "weight")
    let parts: Vec<&str> = key.splitn(2, '.').collect();
    if parts.len() != 2 { return None; }
    let prefix = parts[0];
    let rest = parts[1];
    let did_str = prefix.strip_prefix("domain_")?;
    let did = did_str.parse::<u32>().ok()?;
    // rest 按最后一个 "." 分割 module 和 param name
    let last_dot = rest.rfind('.')?;
    let module = &rest[..last_dot];
    let param = &rest[last_dot + 1..];
    Some((did, module, param))
}
```

## 6.4 每层职责边界 + mod.rs 铁律

| 层级 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| types/ | 零内部依赖 | traits/ utils/ impl/ |
| traits/ | types/ | utils/ impl/ |
| utils/ | types/ | traits/ impl/ |
| impl/ | types/ + utils/ + oyta_router + oyta_tensor + serde + serde_json | 反向依赖自己 |

**所有 mod.rs 只允许**：
```rust
//! impl 层说明

pub mod saver;
pub mod loader;
```

**lib.rs 正确导出**：
```rust
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

pub use types::safetensors_header::SafetensorsHeader;
pub use utils::key_naming::{build_param_key, parse_param_key};
pub use impls::saver::save_sparse_safetensors;
pub use impls::loader::load_sparse_safetensors;

pub mod prelude {
    pub use crate::{save_sparse_safetensors, load_sparse_safetensors, build_param_key, parse_param_key};
    pub use oyta_router::SparseParameterPool;
}
```

---

# 七、验收标准（扩充版：含四层分层 + 完整 safetensors 格式）

```
## 基础验收（原有）
[ ] cargo check --package oyta-serialize 零警告
[ ] Cargo.toml 只有 oyta-tensor = "0.1.1" + safetensors + workspace path（无 rust-version）
[ ] load_sparse_safetensors 加载后 Parameter.data 逐元素和保存前相等
[ ] shape 严格校验（不匹配返回 error）
[ ] allow_partial = false + 文件缺某个参数 → error
[ ] allow_partial = true + 文件只含部分领域 → 加载成功，缺的参数保持初始化值
[ ] active_only = Some(&[1, 2]) → 只保存领域 1 和 2，省磁盘
[ ] 每个函数有中文注释

## 追加：safetensors 格式完整性验收
[ ] save_sparse_safetensors 生成的文件：
    → 8 bytes u64 LE header_len ✓
    → header_len 是 JSON 文本长度（不含 8 字节对齐 padding）✓
    → header JSON 后填充到 8 字节对齐 ✓
    → binary data 按 header.tensors 顺序依次拼接 ✓
    → data_offsets 是相对 header 结束位置的偏移 ✓
    → 每个 tensor 的 dtype="F32"、shape 正确、offsets 正确 ✓
    → 包含 "__metadata__" key（oyta_version）✓
[ ] load_sparse_safetensors 正确：
    → 先读 header_len → 再读 header_len 字节 JSON → 跳过 padding → 读 binary ✓
    → 按 key 匹配 Parameter.name ✓
    → bytes → f32 Vec（little-endian）→ Tensor → 覆盖 Parameter.data ✓
    → allow_partial 行为正确（缺参数报错 vs 跳过）✓

## 追加：四层分层验收
[ ] src/ 目录已改为 types/ + traits/ + utils/ + impl/ 四层结构
[ ] 所有 mod.rs 只含 pub mod xxx; 声明（零 pub use、零实现代码、零 use 语句）
[ ] lib.rs 从子文件直接 pub use
[ ] SafetensorsHeader + TensorEntry 归入 types/ 层
[ ] encoding + alignment + key_naming 纯工具归入 utils/ 层
[ ] save_sparse_safetensors + load_sparse_safetensors 归入 impl/ 层
[ ] 依赖方向单向：types → utils → impls
```

---

# oyta-frame（最后一份）

> Cargo.toml：全部 10 个 crate 依赖 oyta-tensor + oyta-content + ... + oyta-serialize
> src/lib.rs：全部 crate 的 re-export + prelude! 宏
> src/main.rs（顶层 bin）：端到端示例入口

```rust
// oyta-frame/src/lib.rs
//! oyta-frame：一行导入的公共 API 顶层 crate
pub use oyta_tensor;
pub use oyta_content; pub use oyta_router; pub use oyta_expert;
pub use oyta_fusion; pub use oyta_autograd; pub use oyta_loss;
pub use oyta_optim; pub use oyta_trainer; pub use oyta_serialize;

// ===== 最常用类型的便捷 re-export =====
pub use oyta_content::{Domain, RawInput, ContentAnalysis, ContentAnalyzer};
pub use oyta_router::{RoutingStrategy, SparseMaskSet, DomainRouter, Parameter, SparseParameterPool};
pub use oyta_expert::{DomainExpert, ExpertExecutor, MLPExpert, CNN1DExpert, RNNExpert, TransformerExpert};
pub use oyta_fusion::{FusionStrategy, FeatureFusion, ClassificationHead};
pub use oyta_loss::{LossFn, CrossEntropyLoss, MultiDomainLoss};
pub use oyta_optim::{Optimizer, SparseAdam, Scheduler, WarmupCosineAnneal};
pub use oyta_trainer::{SparseTrainer, SparsityConfig, TrainerConfig, SimpleDataLoader, LogCallback};
pub use oyta_serialize::{save_sparse_safetensors, load_sparse_safetensors};
pub use oyta_tensor::{Tensor, Backend};
pub use oyta_tensor::backend::Cpu;

// ===== prelude! 一行导入 =====
pub mod prelude {
    pub use crate::{
        // 底层
        Tensor, Backend, Cpu,
        // Content
        Domain, RawInput, ContentAnalysis, ContentAnalyzer,
        // Router
        RoutingStrategy, SparseMaskSet, DomainRouter, Parameter, SparseParameterPool,
        // Expert
        DomainExpert, ExpertExecutor, MLPExpert, CNN1DExpert, RNNExpert, TransformerExpert,
        // Fusion
        FusionStrategy, FeatureFusion, ClassificationHead,
        // Loss
        LossFn, CrossEntropyLoss, MultiDomainLoss,
        // Optim
        Optimizer, SparseAdam, Scheduler, WarmupCosineAnneal,
        // Trainer
        SparseTrainer, SparsityConfig, TrainerConfig, SimpleDataLoader, LogCallback,
        // Serialize
        save_sparse_safetensors, load_sparse_safetensors,
    };
}
```

---

# 六、全部开发文档最终验收

```
[ ] frame/ 目录下 10 份完整开发文档（00~10）全部创建完成
[ ] 每份 Cargo.toml：
    ✅  oyta-tensor = "0.1.1"（registry 版本）
    ✅  无 path = "/root/oyta_ai/..." 本地依赖
    ✅  无 rust-version = "1.85"（用户项目没指定）
    ✅  只写实际需要的依赖，不重复造轮子
[ ] 所有底层算子调用 oyta_tensor 真实 API：
    ✅  Linear::forward → Tensor::matmul + Tensor::add
    ✅  LayerNorm::forward → Tensor::layernorm
    ✅  TransformerExpert 引用 flash_attention_multihead（oyta_ops）
    ✅  ExpertExecutor.forward_sparse → 框架层 continue 跳过未激活专家
    ✅  SparseParameterPool.zero_grad_sparse / step_sparse / clip_grad_norm_sparse
    ✅  SparseAdam 按领域独立 HashMap<u32, DomainState{m,v,t}>
    ✅  autograd sparse_backward domain_id 过滤整子图跳过
    ✅  autograd Conv1dMasked / BatchNorm1dMasked 完整实现 + L2 mask
    ✅  Conv1dWithSequenceMask L3 padding 清零
    ✅  saver / loader safetensors 完整二进制格式
    ✅  MultiDomainLoss 只对 active_ids 内领域加辅助损失
[ ] 每个函数有中文注释
```
