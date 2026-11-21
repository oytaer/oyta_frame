//! SparseAdam：按领域独立维护优化器状态
//!
//! 为什么按领域独立？
//!   稀疏训练下每个 batch 只激活 K << total 个领域
//!   未激活领域的参数没被 backward → m/v/t 不应该动
//!   如果全局共享 t → 每次 step_sparse 都 +1，但只更新了 K/total 的参数
//!   → 一阶/二阶矩的衰减系数在不同领域上的"有效步数"不一致
//!   → 按领域独立 t 保证每个被激活领域的 Adam 状态正确累积
//!
//! 数学公式（标准 Adam + 解耦 Weight Decay，按领域独立 t）：
//!   m_{t+1} = β1 · m_t + (1-β1) · grad
//!   v_{t+1} = β2 · v_t + (1-β2) · grad²
//!   m̂ = m_{t+1} / (1 - β1^{t+1})       // 偏差校正（关键！不然初始步噪声大）
//!   v̂ = v_{t+1} / (1 - β2^{t+1})
//!   update = lr · m̂ / (√v̂ + ε)
//!   wd_update = lr · wd · param          // 解耦 weight decay（AdamW 风格）
//!   param -= update + wd_update

use std::collections::HashMap;
use oyta_router::SparseParameterPool;
use oyta_tensor::prelude::*;
use crate::traits::optimizer::Optimizer;
use crate::types::domain_state::DomainState;

/// SparseAdam：按领域独立维护 HashMap<u32, DomainState>
pub struct SparseAdam {
    /// 学习率
    lr: f32,
    /// 一阶矩衰减（default 0.9）
    beta1: f32,
    /// 二阶矩衰减（default 0.999）
    beta2: f32,
    /// 数值稳定项（default 1e-8）
    eps: f32,
    /// 解耦 Weight Decay（default 0.0，AdamW 风格）
    wd: f32,
    /// 每个领域独立的优化器状态（⚠️ 关键：不全局共享）
    states: HashMap<u32, DomainState>,
}

impl SparseAdam {
    /// 构造 SparseAdam（默认超参：b1=0.9, b2=0.999, eps=1e-8, wd=0.0）
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.0, states: HashMap::new() }
    }

    /// 设置 beta1/beta2（链式调用）
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self { self.beta1 = b1; self.beta2 = b2; self }

    /// 设置解耦 weight decay（链式调用）
    pub fn with_wd(mut self, wd: f32) -> Self { self.wd = wd; self }

    /// 暴露单个领域的 DomainState（debug / checkpoint 用）
    pub fn domain_state(&self, did: u32) -> Option<&DomainState> { self.states.get(&did) }
}

impl Optimizer for SparseAdam {
    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }

    fn step_sparse(&mut self, active: &[u32], pool: &mut SparseParameterPool) {
        let lr = self.lr;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let wd = self.wd;

        for &did in active {
            // ========== 每个领域独立 t ==========
            let state = self.states.entry(did).or_insert_with(DomainState::new);
            state.t += 1; // 该领域被激活并 step → 计数器 +1
            let t = state.t;

            // bias correction 系数（⚠️ 用该领域自己的 t！不是全局 step 数）
            let b1_pow = b1.powi(t as i32);
            let b2_pow = b2.powi(t as i32);

            let params = pool.domain_params_mut(did);
            for p in params {
                // 没有梯度 → 跳过
                let grad = match p.grad() {
                    Some(g) if p.requires_grad => g.clone(),
                    _ => continue,
                };

                // 确保 m/v 缓存存在
                state.ensure(&p.name, grad.numel());

                // ===== 逐元素 Adam 更新 =====
                let gd = grad.data_slice();
                let m = state.m.get_mut(&p.name).unwrap();
                let v = state.v.get_mut(&p.name).unwrap();

                // 读 Parameter 数据（oyta_tensor 没有 data_slice_mut → 重建模式）
                let old_data = p.data.lock().unwrap();
                let dd = old_data.data_slice();
                let numel = gd.len();
                let mut new_data: Vec<f32> = Vec::with_capacity(numel);

                for i in 0..numel {
                    let g = gd[i];
                    let mut d = dd[i]; // 旧值拷贝

                    // 一阶矩更新
                    m[i] = b1 * m[i] + (1.0 - b1) * g;

                    // 二阶矩更新
                    let g_sq = g * g;
                    v[i] = b2 * v[i] + (1.0 - b2) * g_sq;

                    // 偏差校正（关键！不然初始步噪声大）
                    let m_hat = m[i] / (1.0 - b1_pow);
                    let v_hat = v[i] / (1.0 - b2_pow);

                    // AdamW 更新公式
                    let update = lr * m_hat / (v_hat.sqrt() + eps);
                    let wd_update = lr * wd * d; // 解耦 weight decay
                    d -= update + wd_update;

                    new_data.push(d);
                }
                drop(old_data); // 锁提前释放（arc_mutex）

                // 重建 Tensor 写回（oyta_tensor 没有 data_slice_mut → 必须重建）
                let shape = p.data.lock().unwrap().shape().dims.clone();
                let new_tensor = Tensor::<f32, Cpu>::from_data(new_data, shape).unwrap();
                *p.data.lock().unwrap() = new_tensor;
            }
        }
        // ⚠️ 函数结束时 state &mut borrow 自动 drop → 下次循环 entry 可重入
        // ⚠️ 未激活领域完全不动：它们的 DomainState.t 不变、参数值不变、m/v 不变
    }
}
