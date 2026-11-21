//! key_naming：domain_{did}.{module}.{param} 命名规则
//!
//! 与 oyta_router::param::Parameter.name 完全对齐
//! 例：domain_1.mlp.hidden.weight / domain_2.cnn1d.bn1.gamma

/// 构造 Parameter key（domain_{did}.{module}.{param}）
pub fn build_param_key(did: u32, module: &str, param: &str) -> String {
    format!("domain_{}.{}.{}", did, module, param)
}

/// 解析 Parameter key → (did, module, param)
///
/// 例："domain_1.mlp.hidden.weight" → (1, "mlp.hidden", "weight")
pub fn parse_param_key(key: &str) -> Option<(u32, String, String)> {
    // 按第一个 "." 分割
    let parts: Vec<&str> = key.splitn(2, '.').collect();
    if parts.len() != 2 { return None; }
    let prefix = parts[0];
    let rest = parts[1];

    // 前缀必须是 "domain_{did}"
    let did_str = prefix.strip_prefix("domain_")?;
    let did = did_str.parse::<u32>().ok()?;

    // rest 按最后一个 "." 分割 module 和 param
    let last_dot = rest.rfind('.')?;
    let module = rest[..last_dot].to_string();
    let param = rest[last_dot + 1..].to_string();

    Some((did, module, param))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_parse() {
        let key = build_param_key(1, "mlp.hidden", "weight");
        assert_eq!(key, "domain_1.mlp.hidden.weight");
        let (did, module, param) = parse_param_key(&key).unwrap();
        assert_eq!(did, 1);
        assert_eq!(module, "mlp.hidden");
        assert_eq!(param, "weight");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_param_key("not_a_domain").is_none());
        assert!(parse_param_key("domain_x.foo.bar").is_none()); // did 不是数字
        assert!(parse_param_key("domain_1").is_none()); // 缺 module.param
        assert!(parse_param_key("domain_1.foo").is_none()); // 缺最后一个点
    }
}
