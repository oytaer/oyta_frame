//! Parameter：单个可训练参数包装
//!
//! data 用 oyta_tensor::Tensor<f32, Cpu>（不是自己手搓）
//! Arc<Mutex<Tensor>> 让 Parameter 可 clone + 多线程安全
//!
//! Parameter.name 格式 = "domain_{did}.{module}.{param}"（序列化用）
//!   例: "domain_1.mlp.hidden.weight" / "domain_3.layernorm.gamma"

use std::sync::{Arc, Mutex};
use oyta_tensor::prelude::*;

/// 可训练参数（单个 Parameter 代表一个 Tensor）
#[derive(Clone)]
pub struct Parameter {
    /// 参数值（Arc<Mutex<Tensor<f32, Cpu>>> 多线程安全）
    pub data: Arc<Mutex<Tensor<f32, Cpu>>>,
    /// 梯度（None = 还没 backward 或已 zero_grad）
    pub grad: Option<Tensor<f32, Cpu>>,
    /// 是否需要梯度（冻结参数 = false，训练时完全不参与）
    pub requires_grad: bool,
    /// 命名格式: "domain_{did}.{module}.{param}"（safetensors 序列化用）
    pub name: String,
}

impl Parameter {
    /// 构造：指定初始 Tensor 值 + 完整参数名
    pub fn new(data: Tensor<f32, Cpu>, name: impl Into<String>) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            grad: None,
            requires_grad: true,
            name: name.into(),
        }
    }

    /// 冻结参数（requires_grad = false，训练时不参与 backward）
    pub fn freeze(mut self) -> Self {
        self.requires_grad = false;
        self
    }

    /// 清零梯度（backward 前必须调）
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = None;
        }
    }

    /// 是否已有梯度（None = 还没 backward）
    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// 获取梯度只读引用
    pub fn grad(&self) -> Option<&Tensor<f32, Cpu>> {
        self.grad.as_ref()
    }

    /// 获取梯度可变引用（optimizer step 写回）
    pub fn grad_mut(&mut self) -> Option<&mut Tensor<f32, Cpu>> {
        self.grad.as_mut()
    }

    /// 设置梯度值（sparse_backward 后调用）
    pub fn set_grad(&mut self, g: Tensor<f32, Cpu>) {
        if self.requires_grad {
            self.grad = Some(g);
        }
    }

    /// 梯度累加（梯度累积跨 batch 用，shape 必须一致）
    pub fn accumulate_grad(&mut self, new_grad: &Tensor<f32, Cpu>) {
        if !self.requires_grad {
            return;
        }
        match &mut self.grad {
            Some(existing) => {
                let ed = existing.data_slice();
                let nd = new_grad.data_slice();
                let sum: Vec<f32> = ed.iter().zip(nd.iter()).map(|(a, b)| a + b).collect();
                let shape = existing.shape().dims.clone();
                *existing = Tensor::<f32, Cpu>::from_data(sum, shape).expect("grad 累加构造失败");
            }
            None => {
                self.grad = Some(new_grad.clone());
            }
        }
    }

    /// 参数元素数量（参数量统计用）
    pub fn numel(&self) -> usize {
        self.data.lock().unwrap().numel()
    }
}

/// 按命名约定构造 Parameter（框架层统一入口）
///
/// 命名格式 = "domain_{domain_id}.{module}.{param}"
///   例: build_param(1, "mlp.hidden", "weight", tensor)
///       → name = "domain_1.mlp.hidden.weight"
pub fn build_param(domain_id: u32, module: &str, name: &str, data: Tensor<f32, Cpu>) -> Parameter {
    let full = format!("domain_{}.{}.{}", domain_id, module, name);
    Parameter::new(data, full)
}
