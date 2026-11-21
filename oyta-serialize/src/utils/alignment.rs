//! alignment：8 字节对齐填充（safetensors header 格式要求）
//!
//! safetensors 规范：JSON header 必须填充到 8 字节对齐

/// 计算填充字节数：(8 - len % 8) % 8
pub fn compute_padding(len: usize) -> usize {
    (8 - (len % 8)) % 8
}

/// 把 bytes 填充到 8 字节对齐
pub fn pad_to_align(bytes: &mut Vec<u8>) {
    let pad = compute_padding(bytes.len());
    bytes.extend(std::iter::repeat(b' ').take(pad));
}

/// 去掉尾部 padding（返回 JSON 文本）
pub fn strip_padding(bytes: &[u8]) -> &[u8] {
    // 找到第一个非 padding 位置（从后往前）
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b' ' { end -= 1; }
    &bytes[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        assert_eq!(compute_padding(0), 0);
        assert_eq!(compute_padding(1), 7);
        assert_eq!(compute_padding(7), 1);
        assert_eq!(compute_padding(8), 0);
        assert_eq!(compute_padding(9), 7);
        assert_eq!(compute_padding(15), 1);
        assert_eq!(compute_padding(16), 0);
    }

    #[test]
    fn test_pad_and_strip() {
        let mut v = b"hello".to_vec();
        pad_to_align(&mut v);
        assert_eq!(v.len(), 8);
        assert_eq!(strip_padding(&v), b"hello");
    }

    #[test]
    fn test_already_aligned() {
        let mut v = b"12345678".to_vec();
        pad_to_align(&mut v);
        assert_eq!(v.len(), 8); // 不变
    }
}
