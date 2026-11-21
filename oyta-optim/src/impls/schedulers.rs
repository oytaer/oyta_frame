//! Scheduler 实现：WarmupCosineAnneal / CosineAnneal / StepLR

use crate::traits::optimizer::Optimizer;
use crate::traits::scheduler::Scheduler;

// ==========================================
// WarmupCosineAnneal（推荐默认）
// ==========================================

/// Warmup + Cosine Anneal 组合调度器
///
/// 前 warmup 步：线性从 min_lr 升到 max_lr
/// 之后：余弦退火从 max_lr 降到 min_lr
///
/// 数学公式（cosine annealing 变体）：
///   cos = (step - warmup) / (total - warmup) × π
///   lr = min_lr + 0.5 × (max_lr - min_lr) × (cos + 1.0)
pub struct WarmupCosineAnneal {
    /// warmup 步数（前 warmup 步线性升 lr）
    warmup: u64,
    /// 总训练步数
    total: u64,
    /// 峰值 lr
    max_lr: f32,
    /// 最低 lr
    min_lr: f32,
    /// 当前 step（0 起始）
    step: u64,
    /// 当前 lr（用于 get_lr）
    current: f32,
}

impl WarmupCosineAnneal {
    pub fn new(warmup: u64, total: u64, max_lr: f32, min_lr: f32) -> Self {
        Self { warmup, total, max_lr, min_lr, step: 0, current: min_lr }
    }
}

impl Scheduler for WarmupCosineAnneal {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let lr = if self.step < self.warmup {
            // 线性 warmup：从 min_lr → max_lr
            let r = self.step as f32 / self.warmup.max(1) as f32;
            self.min_lr + (self.max_lr - self.min_lr) * r
        } else {
            // Cosine Anneal：从 max_lr → min_lr
            let progress = (self.step - self.warmup) as f32 / (self.total - self.warmup).max(1) as f32;
            let cos = (progress * std::f32::consts::PI).cos();
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (cos + 1.0)
        };
        self.current = lr.max(self.min_lr);
        opt.set_lr(self.current);
    }

    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}

// ==========================================
// CosineAnneal（纯余弦退火，无 warmup）
// ==========================================

/// 纯 Cosine Anneal（无 warmup）
pub struct CosineAnneal {
    total: u64, max_lr: f32, min_lr: f32, step: u64, current: f32,
}

impl CosineAnneal {
    pub fn new(total: u64, max_lr: f32, min_lr: f32) -> Self {
        Self { total, max_lr, min_lr, step: 0, current: max_lr }
    }
}

impl Scheduler for CosineAnneal {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let progress = self.step as f32 / self.total.max(1) as f32;
        let cos = (progress * std::f32::consts::PI).cos();
        let lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (cos + 1.0);
        self.current = lr.max(self.min_lr);
        opt.set_lr(self.current);
    }
    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}

// ==========================================
// StepLR（等间隔衰减）
// ==========================================

/// StepLR：每 step_size 步 lr 乘 gamma
pub struct StepLR {
    /// 初始 lr
    initial: f32,
    /// 衰减间隔步数
    step_size: u64,
    /// 衰减系数（0.1 = 每 step_size 步 lr × 0.1）
    gamma: f32,
    step: u64,
    current: f32,
}

impl StepLR {
    pub fn new(initial: f32, step_size: u64, gamma: f32) -> Self {
        Self { initial, step_size, gamma, step: 0, current: initial }
    }
}

impl Scheduler for StepLR {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.step += 1;
        let decays = self.step / self.step_size;
        self.current = self.initial * self.gamma.powi(decays as i32);
        opt.set_lr(self.current);
    }
    fn get_lr(&self) -> f32 { self.current }
    fn step_count(&self) -> u64 { self.step }
}
