//! SimpleDataLoader：内存 Vec 直接迭代的最简 DataLoader 实现

use oyta_content::RawInput;
use oyta_tensor::prelude::*;
use crate::traits::data_loader::DataLoader;

/// 内存 Vec 数据源（最简 DataLoader，适合 Phase 1 小规模训练）
pub struct SimpleDataLoader {
    /// 训练样本：(原始输入, 目标 Tensor)
    data: Vec<(RawInput, Tensor<f32, Cpu>)>,
}

impl SimpleDataLoader {
    /// 构造空 DataLoader
    pub fn new() -> Self { Self { data: Vec::new() } }

    /// 追加一个样本
    pub fn push(&mut self, raw: RawInput, target: Tensor<f32, Cpu>) {
        self.data.push((raw, target));
    }

    /// 从 Vec 批量构造
    pub fn from_vec(data: Vec<(RawInput, Tensor<f32, Cpu>)>) -> Self { Self { data } }
}

impl Default for SimpleDataLoader {
    fn default() -> Self { Self::new() }
}

impl DataLoader for SimpleDataLoader {
    fn len(&self) -> usize { self.data.len() }

    fn iter(&self) -> Box<dyn Iterator<Item = (RawInput, Tensor<f32, Cpu>)> + '_> {
        Box::new(self.data.iter().map(|(r, t)| (r.clone(), t.clone())))
    }
}
