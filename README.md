# oyta-frame — 稀疏 MoE 训练框架

> 生产环境可用的异步模块化 Rust 稀疏 Mixture-of-Experts 训练框架

## 定位

**oyta-frame 是一个 Rust 编写的稀疏 MoE（混合专家）训练框架库**，不是可执行程序。它为开发者提供：

- 内容分析 → 稀疏路由 → 多专家并行 → 特征融合 → 任务头 → 端到端训练 的完整构建组件
- 三层稀疏过滤机制（框架层 continue 跳过 + autograd 层 domain_id 过滤 + optimizer 层只更新激活领域）
- safetensors v0 格式稀疏参数保存/加载
- 内置 autograd 自动微分引擎（17+ Op 变体，含三个掩码 backward）

开发者只需组装这些组件，即可构建自己的 AI 模型。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         oyta-frame 顶层 Lib crate                        │
│                 use oyta_frame::prelude::*;  一行导入全部                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│  │ oyta-content │──▶│ oyta-router  │──▶│ oyta-expert               │   │
│  │ 内容分析      │   │ 稀疏路由       │   │ 3 种专家 + 框架层跳过      │   │
│  │ 11 测试       │   │ 18 测试       │   │ 16 测试                    │   │
│  └──────────────┘   └──────┬───────┘   └──────────┬───────────────┘   │
│                            │                       │                   │
│                            ▼                       ▼                   │
│                   ┌──────────────────────────────────┐                │
│                   │         oyta-fusion              │                │
│                   │ 3 种融合 + TaskHead 枚举分发      │                │
│                   │ 14 测试                           │                │
│                   └──────────────┬───────────────────┘                │
│                                  │                                     │
│                                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       训练核心层                                │   │
│  │  oyta-autograd │ oyta-loss │ oyta-optim │ oyta-trainer          │   │
│  │  自动微分引擎    │ 损失函数   │ 优化器+调度  │ 端到端主循环         │   │
│  │  11 测试         │ 11 测试    │ 10 测试     │ 7 测试               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐   │
│  │ oyta-serialize            │   │ oyta-tensor（registry 底层）      │   │
│  │ safetensors v0 稀疏序列化  │   │ Tensor + FlashAttention + BN/Dropout│   │
│  │ 13 测试                    │   │                                  │   │
│  └──────────────────────────┘   └──────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

全部 crate: edition = "2024"，122 单元测试全过，cargo check 零警告
```

---

## 9 个 Crate 详细说明

### 1. oyta-content — 内容分析

**职责**：原始输入 → 领域权重分布（哪个领域占比多）

| 模块 | 功能 | 关键公开类型 |
|------|------|-------------|
| RuleBasedAnalyzer | regex 关键词匹配 | `RuleBasedAnalyzer` |
| TFIDFAnalyzer | TF-IDF 余弦相似度 | `TFIDFAnalyzer` |
| HybridAnalyzer | 规则优先 + TFIDF + MiniMLP 语义编码器 | `HybridAnalyzer` |
| Domain | 领域定义（id + 关键词 + 权重阈值） | `Domain` |
| RawInput | 原始输入封装（支持 metadata + 变长 padding） | `RawInput` |

### 2. oyta-router — 稀疏路由 + 参数池

**职责**：根据内容分析结果，从所有领域中选 K 个激活（MoE 核心）

| 模块 | 功能 | 关键公开类型 |
|------|------|-------------|
| DomainRouter | 路由决策主入口（TopK / Threshold） | `DomainRouter` |
| RoutingStrategy | 路由策略枚举 | `RoutingStrategy` |
| RoutingResult | 路由输出（mask + active_indices + normalized_weights） | `RoutingResult` |
| SparseParameterPool | 按领域独立的参数池（核心：zero_grad_sparse / step_sparse 只激活领域） | `SparseParameterPool` |
| Parameter | 可训练参数（Arc\<Mutex\<Tensor\>\> 包装） | `Parameter` |
| SparseMaskSet | 三层稀疏掩码（L1 domain / L2 channel / L3 sequence） | `SparseMaskSet` |

### 3. oyta-expert — 专家模块

**职责**：多专家并行执行，未激活专家被框架层 `continue` 跳过（Layer 1 稀疏）

| 模块 | 功能 | 关键公开类型 |
|------|------|-------------|
| MLPExpert | Linear → LN → GELU → Linear | `MLPExpert` |
| CNN1DExpert | Conv1d → BN → ReLU → Pool → Linear | `CNN1DExpert` |
| TransformerExpert | LN → QKV → FlashAttn → o_proj → 残差 → FFN → proj（7 组参数） | `TransformerExpert` |
| RNNExpert | 骨架实现 | `RNNExpert` |
| ExpertExecutor | 框架层稀疏执行器（`continue` 跳过未激活专家） | `ExpertExecutor` |
| Linear / LayerNorm / RMSNorm / Conv1d / BatchNorm1d / MaxPool1d | 基础算子层 | `Linear`, `BatchNorm1d`... |

### 4. oyta-fusion — 特征融合 + TaskHead

**职责**：把多个激活专家的输出融合成一个统一表示

| 策略 | 公式 | 有可训练参数 |
|------|------|-------------|
| WeightedSum | `Σ w_i · expert_out_i` | ❌ |
| ConcatLinear | `Linear(concat(expert_outs))` | ✅ 1 个 Linear |
| CrossAttention | FlashAttention(Q=expert_out, K/V=所有expert_out) + gating bias 注入 | ✅ QKV + out_proj + gating |

TaskHead 枚举统一分发：`TaskHead::Classification` / `TaskHead::Regression`

### 5. oyta-autograd — 自动微分引擎（稀疏适配）

**核心设计**：计算图每个 Node 带 `domain_id: Option<u32>`，反向传播时三层过滤未激活领域子图：

```
sparse_backward(loss_id, active_ids, nodes):
  1. topo_reverse 得到反向拓扑序
  2. 遍历每个 Node：
     a. node.domain_id ∈ active_ids → 正常 backward
     b. node.domain_id ∉ active_ids → continue 跳过
     c. node.domain_id = None → 不受过滤（Loss / TaskHead 正常计算）
  3. Op.backward_op 内部 Conv1dMasked / BatchNorm1dMasked / Conv1dWithSequenceMask 二次过滤
```

| 类型 | 变体数 | 关键变体 |
|------|--------|---------|
| Op 枚举 | 17+ | Linear, MSE, CrossEntropy, Conv1dMasked, BatchNorm1dMasked, Conv1dWithSequenceMask, Router... |

### 6. oyta-loss — 损失函数

| 损失 | 公式 | 关键特性 |
|------|------|---------|
| CrossEntropyLoss | `-Σ y·log(softmax(logits))` | label_smoothing 可选 |
| MSELoss | `(1/N)·Σ(pred-target)²` | Reduction: Mean / Sum / None |
| MultiDomainLoss | `main_loss + aux_weight·Σ aux_loss(domain_i)` | **按 active_ids 过滤辅助损失**（防御性二次确认） |

### 7. oyta-optim — 优化器 + Scheduler

**核心设计**：Optimizer 按领域独立维护 `HashMap<u32, DomainState>`，每个领域独立 m / v / t。稀疏训练下未激活领域的状态完全不动。

| 优化器 | 独立状态 | 关键特性 |
|--------|---------|---------|
| SparseAdam | 每个领域独立 `m/v/t` | bias correction 用该领域自己的 t，AdamW 解耦 weight decay |
| SparseSGD | 每个领域独立 `v/t` | Momentum + Nesterov 可选 |
| Scheduler | 无 | WarmupCosineAnneal / CosineAnneal / StepLR |

三个稀疏核心方法：
- `zero_grad_sparse(&active, &mut pool)` — 只清激活领域的 grad
- `step_sparse(&active, &mut pool)` — 只更新激活领域的参数 + state
- `clip_grad_norm_sparse(&active, &mut pool, max_norm)` — 只裁剪激活领域

### 8. oyta-trainer — 端到端稀疏训练主循环

**核心**：`SparseTrainer.train_epoch()` 覆盖 11 步完整稀疏训练时序：

```
Step 1: ContentAnalyzer.analyze(&raw)         → ContentAnalysis（每个样本独立）
Step 2: Router.route(&analysis)               → (RoutingResult, SparseMaskSet)
Step 3: active = routing.active_indices        │ force_all_experts_active（debug 开关）
Step 4: optimizer.zero_grad_sparse(&active)    → 只清激活领域的 grad
Step 5: for expert in experts {
          if active.contains(expert.domain_id) → expert.forward()
          else continue → ← 框架层核心跳过点
        }
Step 6: fusion.fuse(&expert_outs, &normalized_weights)
Step 7: task_head.forward(&fused)
Step 8: loss.compute(..., &active, &expert_outs) → MultiDomainLoss 按 active 过滤
Step 9: sparse_backward(loss_id, &active, nodes) → 三层 domain_id 过滤
Step 10: clip_grad_norm_sparse(&active) + step_sparse(&active) + scheduler.step()
Step 11: callbacks.on_step_end + global_step++
```

### 9. oyta-serialize — safetensors v0 格式稀疏序列化

**文件格式**（严格遵守）：
```
[8 bytes u64 LE header_len]
[JSON header（填充到 8 字节对齐）]
[binary data 所有 Tensor 按 header 顺序排列，f32 little-endian]
```

关键特性：
- `save_sparse_safetensors(pool, path, active_only=None)` — active_only 可选只保存激活领域
- `load_sparse_safetensors(path, &mut pool, allow_partial)` — 严格 shape 校验；allow_partial=true 时缺的参数保持原值
- key 命名统一 `domain_{did}.{module}.{param}`
- `__metadata__` + oyta_version

---

## 三层稀疏过滤端到端协同

```
┌───────────────────────────────────────────────────────────────────────┐
│ Layer 1：框架层 continue 跳过                                          │
│   ExpertExecutor.forward_sparse() → if active_ids.contains → forward │
│   SparseTrainer Step 5 → for expert { if active → forward else cont } │
│   Optimizer step_sparse() → 只遍历 active_ids 内的领域               │
│                                                                       │
│ Layer 2：autograd Op 层掩码                                            │
│   Conv1dMasked.backward → inv_mask 后被 mask 通道梯度置零              │
│   BatchNorm1dMasked.backward → 被 mask 特征 gamma/beta 梯度置零        │
│                                                                       │
│ Layer 3：autograd Op 层 padding post-zero                              │
│   Conv1dWithSequenceMask.backward → padding 时间步 grad_x 强制为 0      │
└───────────────────────────────────────────────────────────────────────┘
```

效果：未激活专家的参数在 backward 中根本不被写 → `Parameter.grad` 永远为 None → Optimizer step_sparse 找不到 grad 就不更新 → 未激活专家的参数值和优化状态（m/v/t）**完全不动** → 完美端到端稀疏。

---

## 快速入门

### 依赖配置

```toml
# Cargo.toml
[dependencies]
oyta_frame = "0.1.0"   # workspace 内部 path 依赖，或发布后的 registry 依赖
```

### 一行导入

```rust
use oyta_frame::prelude::*;
```

### 最小化组装示例（伪代码）

```rust
use oyta_frame::prelude::*;

fn main() {
    // 1. 构造内容分析器（4 个预定义领域）
    let analyzer = Box::new(RuleBasedAnalyzer::new(oyta_content::types::domain::default_domains()));
    let feature_dim = analyzer.feature_dim();

    // 2. 构造稀疏路由（TopK=2）
    let router = DomainRouter::new(RoutingStrategy::TopK { k: 2 }, 4);

    // 3. 构造专家（每个绑定唯一 domain_id）
    let expert_dim = 16;
    let experts: Vec<Box<dyn DomainExpert>> = vec![
        Box::new(MLPExpert::new(0, feature_dim, 32, expert_dim)),
        Box::new(MLPExpert::new(1, feature_dim, 32, expert_dim)),
        Box::new(MLPExpert::new(2, feature_dim, 32, expert_dim)),
    ];

    // 4. 构造融合 + 任务头
    let fusion = FeatureFusion::new(FusionStrategy::WeightedSum, 0, expert_dim);
    let task_head: Box<dyn TaskHeadBase> = Box::new(
        TaskHead::Classification(ClassificationHead::new(0, expert_dim, 3))
    );

    // 5. 构造损失 + 优化器
    let loss = MultiDomainLoss::new(Box::new(CrossEntropyLoss::new(3, Reduction::Mean)));
    let optimizer: Box<dyn Optimizer> = Box::new(SparseAdam::new(1e-3));

    // 6. 构造 Trainer 配置
    let sparsity_config = SparsityConfig {
        enable_expert_sparsity: true,
        routing_strategy: RoutingStrategy::TopK { k: 2 },
        enable_channel_sparsity: false,
        enable_sequence_sparsity: true,
        force_all_experts_active: false,
        log_mask_every_n: None,
    };
    let trainer_config = TrainerConfig {
        max_epochs: 10,
        gradient_accumulation_steps: 1,
        max_grad_norm: 1.0,
        seed: 42,
    };

    // 7. 组装 SparseTrainer + 训练
    let mut trainer = SparseTrainer::new(
        analyzer, router, experts, fusion, task_head,
        optimizer, loss, sparsity_config, trainer_config,
    );

    // 8. 注册参数 + 训练 + 保存
    trainer.register_all_params();
    let avg_loss = trainer.train_epoch(&data_loader, &mut callbacks);
    save_sparse_safetensors(&trainer.param_pool, "model.safetensors", None).unwrap();
}
```

---

## 目录结构

```
oyta-frame/                       ← workspace 根，纯 Lib crate
├── Cargo.toml                    ← workspace 统一依赖 + 顶层 re-export
├── LICENSE                       ← SSPL + AGPL + 商用约束
├── README.md                     ← 本文件
├── src/
│   └── lib.rs                    ← 顶层公共 API 入口（94 个符号 + prelude）
├── docs/
│   └── AI框架系统架构设计.md       ← 完整框架设计文档
├── oyta-content/                 ← 4 层分层：types/traits/utils/analyzers
├── oyta-router/                 ← 4 层分层：types/traits/utils/impls
├── oyta-expert/                  ← 4 层分层：types/traits/utils/impls
├── oyta-fusion/                  ← 4 层分层：types/traits/utils/impls
├── oyta-autograd/                ← 4 层分层：types/traits/utils/impls
├── oyta-loss/                    ← 4 层分层：types/traits/utils/impls
├── oyta-optim/                   ← 4 层分层：types/traits/utils/impls
├── oyta-trainer/                 ← 4 层分层：types/traits/utils/impls
└── oyta-serialize/               ← 4 层分层：types/traits/utils/impls
```

所有 crate 遵守统一的四层分层铁律：
- `types/` — 纯数据结构，零业务逻辑
- `traits/` — 纯接口定义，零实现
- `utils/` — 纯工具函数，可复用
- `impls/` — 具体实现
- `mod.rs` — 只 `pub mod xxx;` 声明，禁止 pub use / 实现代码

---

## Rust 版本 & 依赖

| 项目 | 值 |
|------|-----|
| Rust edition | 2024 |
| 底层算子引擎 | oyta-tensor v0.1.1（registry） |
| workspace 统一依赖 | serde 1.0.229 / regex 1.11.1 / rayon 1.12.0 / thiserror 2.0.16 / anyhow 1.0.93 |
| 测试工具 | serial_test 4.0.1（解决全局 training_mode race condition） |
| 所有时间数据 | u64 秒级时间戳 |
| 所有依赖管理 | workspace 统一 `{ workspace = true }` |

---

## 开源协议

详见 [LICENSE](./LICENSE)。

**核心约束**：
- AGPL-3.0 强 Copyleft：修改后必须开源
- SSPL Section 13 扩展：网络服务使用必须公开整个服务栈源码
- 商用实体 禁止闭源商用和 SaaS 化

**允许**：个人学习 / 学术研究 / 开源项目集成 / 企业内部非核心系统使用

---

## 测试统计

| Crate | 测试数 | 状态 |
|-------|--------|------|
| oyta-content | 22 | ✅ |
| oyta-router | 18 | ✅ |
| oyta-expert | 16 | ✅ |
| oyta-fusion | 14 | ✅ |
| oyta-autograd | 11 | ✅ |
| oyta-loss | 11 | ✅ |
| oyta-optim | 10 | ✅ |
| oyta-trainer | 7 | ✅ |
| oyta-serialize | 13 | ✅ |
| **合计** | **122** | ✅ 全过 |

`cargo check --workspace`：零 error、零 warning

---

## 免责声明

oyta-frame 按"现状"提供，不做任何明示或暗示的保证。贡献者不对任何直接、间接、附带、特殊、后果性或惩罚性损害承担责任。使用者自行负责框架的选型、集成、部署和效果验证。
