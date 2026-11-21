//! SparseSGD：稀疏 SGD（Momentum + 可选 Weight Decay）
//!
//! 数学公式（标准 SGD + Nesterov Momentum 可选，按领域独立状态）：
//!   v_{t+1} = momentum · v_t + grad
//!   param -= lr · (v_{t+1} + momentum · (v_{t+1} - v_t))  （Nesterov 可选）
//!   param -= lr · wd · param  （解耦 weight decay）

use std::collections::HashMap;
use oyta_router::SparseParameterPool;
use oyta_tensor::prelude::*;
use crate::traits::optimizer::Optimizer;
use crate::types::domain_state::DomainState;

/// SparseSGD：按领域独立维护 v（一阶动量）
pub struct SparseSGD {
    lr: f32,
    momentum: f32,       // default 0.9
    wd: f32,             // default 0.0（Weight Decay）
    nesterov: bool,      // default true（Nesterov 加速梯度）
    states: HashMap<u32, DomainState>,
}

impl SparseSGD {
    /// 构造 SparseSGD（默认 momentum=0.9, nesterov=true, wd=0.0）
    pub fn new(lr: f32) -> Self {
        Self { lr, momentum: 0.9, wd: 0.0, nesterov: true, states: HashMap::new() }
    }

    /// 设置 momentum（链式调用）
    pub fn with_momentum(mut self, m: f32) -> Self { self.momentum = m; self }

    /// 设置解耦 weight decay（链式调用）
    pub fn with_wd(mut self, wd: f32) -> Self { self.wd = wd; self }

    /// 关闭 Nesterov（普通 SGD+Momentum）
    pub fn no_nesterov(mut self) -> Self { self.nesterov = false; self }
}

impl Optimizer for SparseSGD {
    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }

    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        let lr = self.lr;
        let m = self.momentum;
        let wd = self.wd;
        let nest = self.nesterov;

        for &did in active {
            // 每个领域独立状态
            let state = self.states.entry(did).or_insert_with(DomainState::new);

            let params = pool.domain_params_mut(did);
            for p in params {
                let grad = match p.grad() { Some(g) if p.requires_grad => g.clone(), _ => continue };
                state.ensure(&p.name, grad.numel());

                let gd = grad.data_slice();
                let v = state.m.get_mut(&p.name).unwrap(); // SGD 用 m 字段存 v（一阶动量）

                // 重建模式读数据（oyta_tensor 没有 data_slice_mut）
                let old_data = p.data.lock().unwrap();
                let dd = old_data.data_slice();
                let numel = gd.len();
                let mut new_data: Vec<f32> = Vec::with_capacity(numel);

                for i in 0..numel {
                    let g = gd[i];
                    let mut d = dd[i];

                    // 动量更新
                    let v_old = v[i];
                    v[i] = m * v_old + g;

                    // SGD 更新 + Nesterov + weight decay
                    if nest {
                        // Nesterov: param -= lr * (v_new + m * (v_new - v_old))
                        d -= lr * (v[i] + m * (v[i] - v_old));
                    } else {
                        d -= lr * v[i];
                    }
                    d -= lr * wd * d; // 解耦 weight decay
                    new_data.push(d);
                }
                drop(old_data);

                // 重建写回
                let shape = p.data.lock().unwrap().shape().dims.clone();
                let new_tensor = Tensor::<f32, Cpu>::from_data(new_data, shape).unwrap();
                *p.data.lock().unwrap() = new_tensor;
            }
        }
    }
}
