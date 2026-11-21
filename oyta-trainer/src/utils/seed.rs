//! 训练可复现性：set_seed / get_random_u64
//!
//! Phase 1 简易实现（用 std::time 种子，不依赖外部 rand 库）
//! 后续可换为真正的伪随机数生成器

use std::sync::atomic::{AtomicU64, Ordering};

/// 全局 seed 存储（Atomic 保证线程安全）
static GLOBAL_SEED: AtomicU64 = AtomicU64::new(42);

/// 设置全局随机种子
pub fn set_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::Relaxed);
}

/// 获取当前全局 seed
pub fn get_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::Relaxed)
}

/// 简易伪随机数（xorshift 变体，用于训练可复现性）
pub fn get_random_u64() -> u64 {
    let mut x = GLOBAL_SEED.fetch_add(1, Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
