//! Callback trait：训练过程回调接口（日志 / checkpoint / 评估）

/// 训练过程回调 trait（Send + Sync → 可跨线程）
pub trait Callback: Send + Sync {
    /// 每个 step 结束时（含激活领域列表，可用于 mask 统计）
    fn on_step_end(&mut self, _step: u64, _loss: f32, _active_ids: &[u32]) {}

    /// 每个 epoch 结束时
    fn on_epoch_end(&mut self, _epoch: usize, _avg_loss: f32) {}
}
