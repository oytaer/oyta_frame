//! LogCallback + SaveCallback：Callback trait 的具体实现

use crate::traits::callback::Callback;

/// 日志回调：每 every_n step 打印一次 loss + active_ids
pub struct LogCallback {
    /// 日志频率（每 every_n step 打印一次）
    pub every_n: u32,
}

impl LogCallback {
    pub fn new(every_n: u32) -> Self { Self { every_n } }
}

impl Callback for LogCallback {
    fn on_step_end(&mut self, step: u64, loss: f32, active_ids: &[u32]) {
        if step as u32 % self.every_n == 0 {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let ids = active_ids.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            println!(
                "[oyta-frame] time={} step={} loss={:.6} active=[{}]",
                ts, step, loss, ids
            );
        }
    }
}

/// Checkpoint 回调：每 every_n epoch 打印一次（Phase 1 骨架）
pub struct SaveCallback {
    /// checkpoint 频率（每 every_n epoch）
    pub every_n: usize,
    /// checkpoint 目录（实际 save 留给 oyta-serialize）
    pub dir: String,
}

impl SaveCallback {
    pub fn new(dir: &str, every_n: usize) -> Self {
        Self { every_n, dir: dir.into() }
    }
}

impl Callback for SaveCallback {
    fn on_epoch_end(&mut self, epoch: usize, _avg: f32) {
        if epoch > 0 && epoch % self.every_n == 0 {
            println!("[oyta-frame] epoch {} → checkpoint dir={}", epoch, self.dir);
        }
    }
}
