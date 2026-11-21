//! traits 层：router crate 没有独立 trait 定义
//!
//! RoutingStrategy 是 enum（不是 trait），DomainRouter 是 struct 组装
//! oyta-router 不需要额外 trait 抽象（保持简单）
//! 预留 traits/mod.rs 保持四层分层一致

// 预留：未来如果需要 RouterCallback / EvalHook 等 trait 可以加在这里
