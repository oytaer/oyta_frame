//! DomainExpert trait：所有专家的统一接口
//!
//! 框架层通过 ExpertExecutor 管理所有专家的稀疏 forward
//! 具体专家（MLPExpert、CNN1DExpert 等）都实现这个 trait

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, SparseParameterPool};

/// 所有领域专家的统一接口
///
/// 框架层约定：
///   - ExpertExecutor.forward_sparse 只对 RoutingResult.active_indices 内的专家调 forward
///   - 未激活专家 forward 零调用、不分配中间 Tensor
///   - train() / eval() 切换全局训练态（影响 Dropout / BatchNorm）
pub trait DomainExpert: Send + Sync {
    /// 该专家绑定的领域 ID（对应 RoutingResult.active_indices 比较）
    fn domain_id(&self) -> u32;

    /// 前向传播（框架层只对激活专家调用）
    fn forward(&self, input: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu>;

    /// 该专家的所有可训练参数（Linear/BN/Conv 的 weight+bias 等）
    fn parameters(&self) -> Vec<Parameter>;

    /// 当前是否训练态（影响 BatchNorm 更新 running 统计 / Dropout 是否启用）
    fn is_training(&self) -> bool;

    /// 切换到训练态
    fn train(&mut self) {
        oyta_tensor::set_training_mode(true);
    }

    /// 切换到推理态
    fn eval(&mut self) {
        oyta_tensor::set_training_mode(false);
    }

    /// 输出维度（框架层 FeatureFusion 需要）
    fn output_dim(&self) -> usize;

    /// 输入维度（框架层 sanity check）
    fn input_dim(&self) -> usize;

    /// 注册参数到 SparseParameterPool（框架初始化时统一调一次）
    fn register(&self, pool: &mut SparseParameterPool) {
        pool.register_domain(self.domain_id(), self.parameters());
    }
}
