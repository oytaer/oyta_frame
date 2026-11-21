//! DataLoader trait：训练数据加载接口

use oyta_content::RawInput;
use oyta_tensor::prelude::*;

/// 数据加载器统一 trait（Send → 可跨线程共享）
pub trait DataLoader: Send {
    /// 样本总数
    fn len(&self) -> usize;

    /// 是否空集
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// 迭代器（迭代每个 batch：(原始输入, 目标 Tensor)）
    fn iter(&self) -> Box<dyn Iterator<Item = (RawInput, Tensor<f32, Cpu>)> + '_>;
}
