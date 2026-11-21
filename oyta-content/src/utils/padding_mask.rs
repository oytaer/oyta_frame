//! PaddingMask：变长输入 padding mask 检测工具
//!
//! 变长输入场景：
//!   一批样本长度不同 [28, 31, 25] → padding 到统一长度 32
//!   Conv1d backward 时 padding 位置 (t ≥ actual_len) 的梯度必须清零（L3 稀疏）
//!
//! ContentAnalyzer 在预处理阶段调用这里生成 mask，传给 DomainRouter
//! DomainRouter 再注入 SparseMaskSet.sequence_masks（L3 层级）

pub struct PaddingMask;

impl PaddingMask {
    /// 变长输入检测：返回每个样本的 padding mask
    ///
    /// # 参数
    /// - sample_lengths: 每个样本实际长度（不含 padding）
    ///   例：[28, 31, 25]
    /// - max_length: padding 统一到的长度
    ///   例：32
    ///
    /// # 返回
    /// - Vec<Vec<f32>>：shape = [num_samples, max_length]
    ///   mask[n][t] = 0.0 → 位置 t 是有效内容
    ///   mask[n][t] = 1.0 → 位置 t 是 padding
    ///
    /// # 示例
    ///   sample_lengths = [28, 31, 25], max_length = 32
    ///   mask[0] = [0,0,...,0(28个), 1,1,1,1]  后 4 个 padding
    ///   mask[1] = [0,0,...,0(31个), 1]        后 1 个 padding
    ///   mask[2] = [0,0,...,0(25个), 1,1,1,1,1,1,1]  后 7 个 padding
    pub fn detect(sample_lengths: &[usize], max_length: usize) -> Vec<Vec<f32>> {
        sample_lengths.iter().map(|&actual_len| {
            (0..max_length).map(|t| {
                if t < actual_len { 0.0f32 } else { 1.0f32 }
            }).collect()
        }).collect()
    }

    /// 简化版：所有样本同一长度（全部有效，无 padding）
    /// 返回全零 mask
    pub fn all_valid(num_samples: usize, max_length: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0f32; max_length]; num_samples]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let mask = PaddingMask::detect(&[28, 31, 25], 32);
        assert_eq!(mask.len(), 3);
        assert_eq!(mask[0].len(), 32);
        assert_eq!(mask[0][27], 0.0);  // t=27 有效（0..27 共 28 个）
        assert_eq!(mask[0][28], 1.0);  // t=28 padding
        assert_eq!(mask[0][31], 1.0);  // t=31 padding
        assert_eq!(mask[1][30], 0.0);
        assert_eq!(mask[1][31], 1.0);
        assert_eq!(mask[2][24], 0.0);
        assert_eq!(mask[2][25], 1.0);
        assert_eq!(mask[2][31], 1.0);
    }

    #[test]
    fn test_all_valid() {
        let mask = PaddingMask::all_valid(2, 10);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0].iter().sum::<f32>(), 0.0);
        assert_eq!(mask[1].iter().sum::<f32>(), 0.0);
    }
}
