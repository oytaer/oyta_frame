//! MSELoss：均方误差损失
//!
//! 数学：loss = (1/N) × Σ (pred[i] - target[i])²
//! grad = (2/N) × (pred - target)  （解析梯度）

use oyta_tensor::prelude::*;
use crate::traits::loss_fn::LossFn;
use crate::types::reduction::Reduction;

/// 均方误差损失
pub struct MSELoss {
    pub reduction: Reduction,
}

impl MSELoss {
    pub fn new(reduction: Reduction) -> Self { Self { reduction } }
}

impl LossFn for MSELoss {
    fn compute(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let ld = logits.data_slice();
        let td = target.data_slice();
        let total = ld.len();
        assert_eq!(total, td.len(), "MSELoss pred/target numel 必须相同");

        // per-element squared error
        let sq_err: Vec<f32> = ld.iter().zip(td.iter())
            .map(|(p, t)| (p - t) * (p - t)).collect();

        match self.reduction {
            Reduction::Mean => {
                let m = sq_err.iter().sum::<f32>() / total as f32;
                Tensor::<f32, Cpu>::from_data(vec![m], [1]).unwrap()
            }
            Reduction::Sum => {
                let s: f32 = sq_err.iter().sum();
                Tensor::<f32, Cpu>::from_data(vec![s], [1]).unwrap()
            }
            Reduction::None => {
                let shape = logits.shape().dims.clone();
                Tensor::<f32, Cpu>::from_data(sq_err, shape).unwrap()
            }
        }
    }

    fn grad(&self, logits: &Tensor<f32, Cpu>, target: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        let ld = logits.data_slice();
        let td = target.data_slice();
        let total = ld.len();
        let mut g: Vec<f32> = ld.iter().zip(td.iter())
            .map(|(p, t)| 2.0 * (p - t)).collect();

        if matches!(self.reduction, Reduction::Mean) {
            let scale = 1.0 / total as f32;
            for v in g.iter_mut() { *v *= scale; }
        }
        let shape = logits.shape().dims.clone();
        Tensor::<f32, Cpu>::from_data(g, shape).unwrap()
    }

    fn reduction(&self) -> Reduction { self.reduction }
}
