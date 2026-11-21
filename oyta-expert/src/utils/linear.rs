//! Linear 层：y = x @ W^T + b
//!
//! 全部算子走 oyta_tensor API：
//!   Tensor::matmul（矩阵乘法）
//!   Tensor::add（偏置广播后相加）
//!   Tensor::transpose（转置）
//!
//! 参数初始化：Kaiming 均匀分布
//!   先 randn 生成标准正态 → 乘 scale = sqrt(6 / fan_in)

use oyta_tensor::prelude::*;
use oyta_router::{Parameter, build_param};

/// 全连接层：y = x @ W^T + b
pub struct Linear {
    /// 权重 [out_features, in_features]
    weight: Parameter,
    /// 偏置 [out_features]
    bias: Parameter,
    /// 记录输入输出维度（框架层 sanity check）
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// 构造 Linear（Kaiming 均匀分布自动初始化）
    pub fn new(domain_id: u32, module: &str, in_features: usize, out_features: usize) -> Self {
        // Kaiming 均匀分布 scale = sqrt(6 / fan_in)
        let scale = (6.0 / in_features as f32).sqrt();
        // randn 生成标准正态 → 底层 mul scale（D::elwise_mul Rayon 并行）
        let w_raw = Tensor::<f32, Cpu>::randn([out_features, in_features]);
        let w_tensor = {
            let scale_tensor = Tensor::<f32, Cpu>::full([out_features, in_features], scale);
            w_raw.mul(&scale_tensor).expect("Linear weight scale mul 失败")
        };
        // bias 初始化为 0
        let b_tensor = Tensor::<f32, Cpu>::zeros([out_features]);

        Self {
            weight: build_param(domain_id, module, "weight", w_tensor),
            bias:   build_param(domain_id, module, "bias",   b_tensor),
            in_features,
            out_features,
        }
    }

    /// 前向：y = x @ W^T + b
    ///
    /// x 形状：[batch, in_features]
    /// y 形状：[batch, out_features]
    pub fn forward(&self, x: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // 取出参数值（Arc<Mutex> → clone 一份）
        let w = self.weight.data.lock().unwrap().clone();  // [out, in]
        let b = self.bias.data.lock().unwrap().clone();    // [out]

        // W^T = transpose(w) → [in, out]（直接返回 Self，不是 Result）
        let wt = w.transpose(0, 1);

        // wx = x @ W^T → [batch, in] @ [in, out] = [batch, out]
        let wx = x.matmul(&wt).expect("Linear::forward matmul 失败");

        // 偏置广播：[out] → [batch, out]（每一行加上相同的 bias）
        let wx_shape = wx.shape();
        let batch = wx_shape.dims[0];
        let out_feat = b.shape().dims[0];
        let bd = b.data_slice();
        let mut b_broadcast = Vec::with_capacity(batch * out_feat);
        for _ in 0..batch {
            b_broadcast.extend_from_slice(bd);
        }
        let b_full = Tensor::<f32, Cpu>::from_data(b_broadcast, [batch, out_feat])
            .expect("Linear bias 广播构造失败");

        // y = wx + b_full
        wx.add(&b_full).expect("Linear::forward bias add 失败")
    }

    /// 该层所有可训练参数（weight + bias）
    pub fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}
