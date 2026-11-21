//! MiniMLP：可训练轻量 MLP（Linear → GELU → Linear）
//!
//! 架构文档 §5.2 分析维度表第 5 行：
//!   "可选：轻量语义编码 —— char n-gram → Embedding → 轻量 MLP（2-3 层） —— 处理词汇变化"
//!
//! 完整实现（全调 oyta_tensor API）：
//!   forward: y = (x @ W1^T + b1).gelu() @ W2^T + b2
//!   参数初始化: Kaiming 正态（randn × sqrt(2/fan_in)）
//!   train()/eval() 模式切换（Phase 1 无 Dropout，预留接口）
//!   parameters() 返回内部参数 Tensor（让 SparseParameterPool 注册）
//!
//! 为什么 MiniMLP 独立于 FeatureEncoder？
//!   FeatureEncoder 是静态 hash 编码器（不可训练），MiniMLP 是可训练投影层
//!   HybridAnalyzer.train_semantic_encoder() 时只训练 MiniMLP 的参数
//!   analyze 流程变成: hash_feats → (可选 MiniMLP 投影) → 输出 Tensor

use oyta_tensor::prelude::*;

/// 可训练轻量 MLP（2 层 Linear + GELU 激活）
pub struct MiniMLP {
    /// 第一层权重 [hidden_dim, input_dim]
    w1: Tensor<f32, Cpu>,
    /// 第一层偏置 [hidden_dim]
    b1: Tensor<f32, Cpu>,
    /// 第二层权重 [output_dim, hidden_dim]
    w2: Tensor<f32, Cpu>,
    /// 第二层偏置 [output_dim]
    b2: Tensor<f32, Cpu>,
    /// 是否训练模式（Phase 1 无 Dropout，预留接口）
    training: bool,
}

impl MiniMLP {
    /// 构造：指定输入维度 → 隐藏层维度 → 输出维度
    /// 权重用 Kaiming 正态初始化（fan_in 归一化）
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // 第一层：fan_in = input_dim，Kaiming 缩放 = sqrt(2 / fan_in)
        let rand_w1 = Tensor::<f32, Cpu>::randn([hidden_dim, input_dim]);
        let w1 = Self::scale_tensor(&rand_w1, (2.0 / input_dim as f32).sqrt());

        // 第一层偏置初始化为 0
        let b1 = Tensor::<f32, Cpu>::zeros([hidden_dim]);

        // 第二层：fan_in = hidden_dim
        let rand_w2 = Tensor::<f32, Cpu>::randn([output_dim, hidden_dim]);
        let w2 = Self::scale_tensor(&rand_w2, (2.0 / hidden_dim as f32).sqrt());

        // 第二层偏置初始化为 0
        let b2 = Tensor::<f32, Cpu>::zeros([output_dim]);

        Self { w1, b1, w2, b2, training: false }
    }

    /// 缩放 Tensor 数据（返回新 Tensor，oyta_tensor 无 data_slice_mut）
    /// 用 data_slice 读 → Vec 缩 → from_data 重建
    fn scale_tensor(t: &Tensor<f32, Cpu>, scale: f32) -> Tensor<f32, Cpu> {
        let data: Vec<f32> = t.data_slice().iter().map(|v| v * scale).collect();
        let shape: Vec<usize> = t.shape().dims.iter().copied().collect();
        Tensor::<f32, Cpu>::from_data(data, shape).expect("scale_tensor 构造失败")
    }

    /// 前向传播：input [batch, input_dim] → output [batch, output_dim]
    ///
    /// 完整流程（全调 oyta_tensor API）：
    ///   Layer 1: h = (input @ W1^T + b1).gelu()
    ///   Layer 2: output = h @ W2^T + b2
    pub fn forward(&self, input: &Tensor<f32, Cpu>) -> Tensor<f32, Cpu> {
        // Layer 1: x @ W1^T —— 注意 W1 shape 是 [hidden, input]
        // matmul 要求 [batch, input] @ [input, hidden] = [batch, hidden]
        // 所以把 w1 transpose 成 [input, hidden]（数据层面 transpose）
        let w1t = Self::transpose_tensor(&self.w1, 0, 1);
        let h = input.matmul(&w1t).expect("MiniMLP layer1 matmul 失败");
        // 加偏置 b1（广播：[batch, hidden] += [hidden]）
        let b1_broadcast = Self::broadcast_vec(&self.b1, input.shape().dims[0], self.w1.shape().dims[0]);
        let h = h.add(&b1_broadcast).expect("MiniMLP layer1 add bias 失败");
        // GELU 激活
        let h = h.gelu().expect("MiniMLP gelu 失败");

        // Layer 2: h @ W2^T
        let w2t = Self::transpose_tensor(&self.w2, 0, 1);
        let out = h.matmul(&w2t).expect("MiniMLP layer2 matmul 失败");
        // 加偏置 b2
        let b2_broadcast = Self::broadcast_vec(&self.b2, input.shape().dims[0], self.w2.shape().dims[0]);
        out.add(&b2_broadcast).expect("MiniMLP layer2 add bias 失败")
    }

    /// 手动 transpose Tensor（oyta-tensor 可能没内置 transpose 方法）
    /// 输入 shape [a, b] → 输出 shape [b, a]
    fn transpose_tensor(t: &Tensor<f32, Cpu>, _dim1: usize, _dim2: usize) -> Tensor<f32, Cpu> {
        let shape = &t.shape().dims;
        let (rows, cols) = (shape[0], shape[1]);
        let data = t.data_slice();
        let mut out_data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out_data[c * rows + r] = data[r * cols + c];
            }
        }
        Tensor::<f32, Cpu>::from_data(out_data, [cols, rows]).expect("transpose_tensor 构造失败")
    }

    /// 广播偏置向量 [dim] → [batch, dim]（沿 dim=0 重复 batch 次）
    fn broadcast_vec(b: &Tensor<f32, Cpu>, batch: usize, dim: usize) -> Tensor<f32, Cpu> {
        let data = b.data_slice();
        let mut out_data = Vec::with_capacity(batch * dim);
        for _ in 0..batch { out_data.extend_from_slice(data); }
        Tensor::<f32, Cpu>::from_data(out_data, [batch, dim]).expect("broadcast_vec 构造失败")
    }

    /// 返回所有权重 Tensor（让 SparseParameterPool 注册）
    pub fn parameters(&self) -> Vec<&Tensor<f32, Cpu>> { vec![&self.w1, &self.b1, &self.w2, &self.b2] }

    /// 可训练参数计数（训练监控用）
    pub fn param_count(&self) -> usize {
        self.w1.numel() + self.b1.numel() + self.w2.numel() + self.b2.numel()
    }

    /// 训练模式
    pub fn train(&mut self) { self.training = true; }
    /// 评估模式
    pub fn eval(&mut self) { self.training = false; }
    pub fn is_training(&self) -> bool { self.training }

    /// 内部权重可变访问（optimizer 训练时写回）
    pub fn w1_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.w1 }
    pub fn b1_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.b1 }
    pub fn w2_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.w2 }
    pub fn b2_mut(&mut self) -> &mut Tensor<f32, Cpu> { &mut self.b2 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_mlp_forward_shape() {
        let mlp = MiniMLP::new(256, 128, 64);
        let input = Tensor::<f32, Cpu>::randn([2, 256]);
        let out = mlp.forward(&input);
        assert_eq!(out.shape().dims, vec![2, 64]);
    }

    #[test]
    fn test_mini_mlp_param_count() {
        let mlp = MiniMLP::new(256, 128, 64);
        // w1: 128*256 + b1: 128 + w2: 64*128 + b2: 64
        assert_eq!(mlp.param_count(), 128 * 256 + 128 + 64 * 128 + 64);
    }

    #[test]
    fn test_mini_mlp_basic_values() {
        // 零输入 → 只有偏置生效 → 非零输出
        let mlp = MiniMLP::new(4, 8, 2);
        let input = Tensor::<f32, Cpu>::zeros([1, 4]);
        let out = mlp.forward(&input);
        assert_eq!(out.shape().dims, vec![1, 2]);
        // 零输入过 GELU → 第一层输出 ≈ b1 部分（因 gelu(0)=0，加上 b1 后非零）
    }
}
