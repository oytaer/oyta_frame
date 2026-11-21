//! Task Heads：ClassificationHead + RegressionHead
//!
//! 统一实现 TaskHeadBase trait，Linear 复用 oyta_expert

use oyta_tensor::prelude::*;
use oyta_expert::Linear;
use oyta_router::Parameter;
use crate::traits::task_head::TaskHeadBase;

/// 分类头：Linear → logits（Softmax 在 Loss 内部做）
pub struct ClassificationHead {
    /// 输出投影 Linear
    linear: Linear,
    /// 类别数
    num_classes: usize,
}

impl ClassificationHead {
    pub fn new(did: u32, fused_dim: usize, num_classes: usize) -> Self {
        Self {
            linear: Linear::new(did, "task_head", fused_dim, num_classes),
            num_classes,
        }
    }
}

impl TaskHeadBase for ClassificationHead {
    fn forward(&self, fused: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // 直接输出 logits，Softmax 在 Loss 内部做
        self.linear.forward(fused)
    }
    fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
    fn output_dim(&self) -> usize { self.num_classes }
}

/// 回归头：Linear → 标量
pub struct RegressionHead {
    /// 输出投影 Linear
    linear: Linear,
}

impl RegressionHead {
    pub fn new(did: u32, fused_dim: usize) -> Self {
        Self { linear: Linear::new(did, "task_head", fused_dim, 1) }
    }
}

impl TaskHeadBase for RegressionHead {
    fn forward(&self, fused: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        self.linear.forward(fused)
    }
    fn parameters(&self) -> Vec<Parameter> { self.linear.parameters() }
    fn output_dim(&self) -> usize { 1 }
}

/// TaskHead 枚举：统一分发到 ClassificationHead / RegressionHead
///
/// 框架层 TaskHeadBase trait 要求实现 forward + parameters + output_dim
/// 匹配 FeatureFusion 的三选一 Option 模式，TaskHead 用 enum match 分发
pub enum TaskHead {
    Classification(ClassificationHead),
    Regression(RegressionHead),
}

impl TaskHeadBase for TaskHead {
    fn forward(&self, fused: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        match self {
            TaskHead::Classification(h) => h.forward(fused),
            TaskHead::Regression(h)     => h.forward(fused),
        }
    }
    fn parameters(&self) -> Vec<Parameter> {
        match self {
            TaskHead::Classification(h) => h.parameters(),
            TaskHead::Regression(h)     => h.parameters(),
        }
    }
    fn output_dim(&self) -> usize {
        match self {
            TaskHead::Classification(h) => h.output_dim(),
            TaskHead::Regression(h)     => h.output_dim(),
        }
    }
}
