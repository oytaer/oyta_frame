//! impls 层：路由策略 + DomainRouter 主入口
//!
//! 为什么叫 impls 而不是 impl？
//!   impl 是 Rust 关键字，不能做模块名
//!   impls 是 implementations 的缩写

pub mod routing_strategy;
pub mod domain_router;
