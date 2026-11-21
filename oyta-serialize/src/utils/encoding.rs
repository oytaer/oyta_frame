//! encoding：bytes ↔ f32 little-endian 编解码（safetensors 格式要求）

/// f32 slice → little-endian bytes Vec
pub fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data { bytes.extend_from_slice(&v.to_le_bytes()); }
    bytes
}

/// little-endian bytes → f32 Vec（bytes 长度必须是 4 的倍数）
pub fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("bytes 长度 {} 不是 4 的倍数", bytes.len()));
    }
    let numel = bytes.len() / 4;
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let arr: [u8; 4] = [bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let v = vec![1.0f32, -2.5, 3.14, 0.0];
        let bytes = f32_slice_to_bytes(&v);
        assert_eq!(bytes.len(), 16);
        let back = bytes_to_f32_vec(&bytes).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_bytes_not_multiple_of_4() {
        let bytes = vec![0u8; 5];
        assert!(bytes_to_f32_vec(&bytes).is_err());
    }
}
