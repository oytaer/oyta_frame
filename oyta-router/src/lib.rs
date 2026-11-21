//! oyta-router：稀疏路由 crate（Phase 2）
//!
//! 职责：稀疏路由策略 + 三层掩码全集 + 按领域独立的 SparseParameterPool
//!
//! 四层分层目录结构：
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ lib.rs              顶层统一 pub mod 声明 + pub use + prelude   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ types/              数据类型层（纯结构体：SparseMaskSet + OpType）│
//! │   ├── mod.rs        只 pub mod 声明（零 pub use / 零实现）      │
//! │   ├── sparse_mask.rs       SparseMaskSet + MaskLevel + build() │
//! │   └── op_type.rs           OpType 枚举（含 3 个掩码变体）        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ traits/             接口层（router crate 无独立 trait，预留）   │
//! │   └── mod.rs        只 pub mod 声明                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ utils/              工具层（参数管理）                           │
//! │   ├── mod.rs        只 pub mod 声明                             │
//! │   ├── param.rs              Parameter + build_param             │
//! │   └── param_pool.rs          SparseParameterPool（zero_grad_sparse / step_sparse）│
//! ├─────────────────────────────────────────────────────────────────┤
//! │ impl/               实现层（路由策略 + DomainRouter 主入口）    │
//! │   ├── mod.rs        只 pub mod 声明                             │
//! │   ├── routing_strategy.rs   RoutingStrategy（TopK / Threshold） │
//! │   └── domain_router.rs      DomainRouter（route 主入口）         │
//! └─────────────────────────────────────────────────────────────────┘

// ===== 四层分层：types → traits → utils → impl =====
pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 公共导出（从子文件直接 pub use，不经过 mod.rs）=====
// types 层
pub use types::sparse_mask::{SparseMaskSet, DomainSparseMask, MaskLevel};
pub use types::op_type::OpType;
// utils 层
pub use utils::param::{Parameter, build_param};
pub use utils::param_pool::SparseParameterPool;
// impl 层
pub use impls::routing_strategy::{RoutingStrategy, RoutingResult};
pub use impls::domain_router::DomainRouter;

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{
        SparseMaskSet, DomainSparseMask, MaskLevel, OpType,
        Parameter, SparseParameterPool,
        RoutingStrategy, RoutingResult, DomainRouter,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_content::ContentAnalysis;
}
