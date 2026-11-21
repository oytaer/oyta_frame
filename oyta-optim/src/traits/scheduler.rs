//! Scheduler trait：学习率调度器接口

use crate::traits::optimizer::Optimizer;

/// 学习率调度器 trait
pub trait Scheduler: Send + Sync {
    /// 执行一步调度：根据当前 step 计算新 lr → 调 optimizer.set_lr()
    fn step(&mut self, opt: &mut dyn Optimizer);

    /// 当前学习率
    fn get_lr(&self) -> f32;

    /// 已执行的 step 数
    fn step_count(&self) -> u64;
}
