//! oyta-trainer：端到端稀疏训练主循环 + DataLoader + Callback + Config
//!
//! 四层分层目录：
//! types/       SparsityConfig + TrainerConfig（纯配置）
//! traits/      DataLoader trait + Callback trait
//! utils/       set_seed / get_random_u64（训练可复现性）
//! impls/       SparseTrainer + SimpleDataLoader + LogCallback + SaveCallback
//!
//! 核心：train_epoch 完整覆盖 Step 1~11 端到端稀疏训练时序
//!   Step 1: ContentAnalyzer.analyze（每个样本独立）
//!   Step 2: Router.route（每个 batch 动态决策 active_ids）
//!   Step 3: force_all_experts_active debug 开关
//!   Step 4: zero_grad_sparse（只清激活领域）
//!   Step 5: for expert in experts → continue 跳过未激活专家 ⚠️ 核心
//!   Step 6: fusion.fuse（只含激活专家输出）
//!   Step 7: task_head.forward
//!   Step 8: loss.compute（MultiDomainLoss 按 active 过滤）
//!   Step 9: sparse_backward（Phase 1 骨架，注释说明）
//!   Step 10: clip_grad_norm_sparse + step_sparse（累积步末尾）
//!   Step 11: callbacks + global_step++
//!
//! 三层稀疏过滤端到端协同：
//!   Layer 1: ExpertExecutor continue 跳过 + sparse_backward domain_id 过滤 + Optimizer 只遍历 active
//!   Layer 2: Conv1dMasked pre-mask / BatchNorm1dMasked post-mask（autograd Op 内部）
//!   Layer 3: Conv1dWithSequenceMask post-zero padding（autograd Op 内部）

pub mod types;
pub mod traits;
pub mod utils;
pub mod impls;

// ===== 顶层导出 =====
pub use types::config::{SparsityConfig, TrainerConfig};
pub use traits::data_loader::DataLoader;
pub use traits::callback::Callback;
pub use utils::seed::{set_seed, get_seed, get_random_u64};
pub use impls::sparse_trainer::SparseTrainer;
pub use impls::simple_data_loader::SimpleDataLoader;
pub use impls::callbacks::{LogCallback, SaveCallback};

// ===== prelude：一行导入全部 =====
pub mod prelude {
    pub use crate::{
        SparsityConfig, TrainerConfig, DataLoader, Callback,
        SparseTrainer, SimpleDataLoader, LogCallback, SaveCallback,
    };
    pub use oyta_tensor::prelude::*;
    pub use oyta_content::RawInput;
}

// ===== 单元测试 =====
#[cfg(test)]
mod tests {
    use super::*;

    // ===== Default 配置验证 =====
    #[test]
    fn test_sparsity_config_default() {
        let sc = SparsityConfig::default();
        assert!(sc.enable_expert_sparsity, "默认应启用专家稀疏");
        assert!(!sc.enable_channel_sparsity, "默认关闭通道稀疏");
        assert!(sc.enable_sequence_sparsity, "默认启用序列稀疏");
        assert!(!sc.force_all_experts_active, "默认关闭 debug 开关");
        assert!(sc.log_mask_every_n.is_none(), "默认不打印 mask 日志");
    }

    #[test]
    fn test_trainer_config_default() {
        let tc = TrainerConfig::default();
        assert_eq!(tc.gradient_accumulation_steps, 1);
        assert_eq!(tc.max_epochs, 10);
        assert!((tc.max_grad_norm - 1.0).abs() < 1e-6);
        assert_eq!(tc.seed, 42);
    }

    // ===== SimpleDataLoader =====
    #[test]
    fn test_simple_data_loader_basic() {
        use oyta_tensor::{Tensor, Cpu};

        let mut dl = SimpleDataLoader::new();
        assert_eq!(dl.len(), 0);
        assert!(dl.is_empty());

        let raw = oyta_content::RawInput::from_text("test content");
        let tgt = Tensor::<f32, Cpu>::from_data(vec![1.0, 0.0], [2]).unwrap();
        dl.push(raw, tgt);

        assert_eq!(dl.len(), 1);
        assert!(!dl.is_empty());

        let items: Vec<_> = dl.iter().collect();
        assert_eq!(items.len(), 1);
    }

    // ===== LogCallback / SaveCallback =====
    #[test]
    fn test_log_callback_construction() {
        let cb = LogCallback::new(10);
        assert_eq!(cb.every_n, 10);
    }

    #[test]
    fn test_save_callback_construction() {
        let cb = SaveCallback::new("/tmp/ckpt", 5);
        assert_eq!(cb.every_n, 5);
        assert_eq!(cb.dir, "/tmp/ckpt");
    }

    // ===== set_seed / get_random_u64 =====
    #[test]
    fn test_seed_and_random() {
        set_seed(12345);
        assert_eq!(get_seed(), 12345);
        let r1 = get_random_u64();
        let r2 = get_random_u64();
        assert_ne!(r1, r2, "连续调用应返回不同值");

        // 复现性：reset 后同 seed → 同序列
        set_seed(12345);
        assert_eq!(get_random_u64(), r1);
        assert_eq!(get_random_u64(), r2);
    }

    // ===== 四层分层单向依赖验证 =====
    // 设计 §8.2 要求：types → traits/utils → impls → 所有其他 crate
    #[test]
    fn test_layers_boundary() {
        // 这是编译时检查：impls::sparse_trainer 可以用所有其他 crate ✅
        // types::config 只能用 oyta-router（RoutingStrategy）✅
        // traits::data_loader 用 oyta-content + oyta-tensor ✅
        // 没有反向依赖（impls 不被 types/traits/utils 依赖）✅
        // 此测试是占位，真正检查在 cargo check 编译器
        let _ = SparsityConfig::default();
        let _ = TrainerConfig::default();
    }
}
