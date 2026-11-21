//! oyta-frame：生产环境可用的稀疏 MoE 训练框架公共 API 入口
//!
//! 本 crate 是整个 oyta 框架的顶层统一入口，职责：
//!   1. re-export 9 个框架 crate + oyta-tensor 底层算子的全部公开类型
//!   2. 提供 `prelude` 模块让用户一行导入全部：`use oyta_frame::prelude::*;`
//!   3. 所有 re-export 不引入新逻辑，纯类型转发，保证零注入点零漏洞
//!
//! 框架架构（完整四层分层）：
//! ```text
//! oyta-tensor    ← 底层算子引擎（registry）
//! oyta-content   ← 内容分析（RuleBased / TFIDF / Hybrid）
//! oyta-router    ← 稀疏路由 + SparseParameterPool
//! oyta-expert    ← 专家模块（MLP / CNN1D / RNN / Transformer）
//! oyta-fusion    ← 特征融合（WeightedSum / ConcatLinear / CrossAttention）+ TaskHead
//! oyta-autograd  ← 自动微分引擎（sparse_backward 三层 domain_id 过滤）
//! oyta-loss      ← 损失函数（CrossEntropy / MSE / MultiDomainLoss）
//! oyta-optim     ← 优化器 + Scheduler（SparseAdam / SparseSGD / WarmupCosineAnneal）
//! oyta-trainer   ← 端到端稀疏训练主循环（SparseTrainer 11 步时序）
//! oyta-serialize ← safetensors v0 格式稀疏参数保存/加载
//! ```

// ===== oyta-tensor：底层算子引擎（所有框架的基础依赖）=====
pub use oyta_tensor::{Tensor, Cpu};

// ===== oyta-content：内容分析 crate =====
pub use oyta_content::{
    Domain,                    // 领域定义 + 预定义金融/医疗/法律/科技领域
    RawInput,                  // 原始输入（支持秒级时间戳 + metadata）
    ContentAnalysis,           // 分析结果（domain_weights + feature_vector + padding_mask + evidence）
    Evidence,                  // 证据记录（关键词/结构化数据命中）
    ContentAnalyzer,           // 内容分析器 trait（Send + Sync）
    LabeledContent,            // 带标注训练数据（用于 HybridAnalyzer 语义编码器训练）
    FeatureEncoder,            // 特征编码器（文本 → 固定维度 Tensor）
    PaddingMask,               // 变长 padding mask 检测工具
    RuleBasedAnalyzer,         // 基于规则关键词匹配的分析器
    TFIDFAnalyzer,             // TF-IDF 余弦相似度分析器
    HybridAnalyzer,            // 混合分析器（规则优先 + TFIDF + MiniMLP 语义编码器）
};

// ===== oyta-router：稀疏路由 + 参数池 crate =====
pub use oyta_router::{
    SparseMaskSet,             // 三层稀疏掩码全集（L1 domain / L2 channel / L3 sequence）
    DomainSparseMask,          // 单个领域的掩码数据
    MaskLevel,                 // 掩码层级枚举（L1 / L2 / L3）
    OpType,                    // 算子类型枚举（含三个掩码变体 + Router）
    Parameter,                 // 可训练参数（Arc<Mutex<Tensor>> + 秒级时间戳 + domain 归属）
    SparseParameterPool,       // 按领域独立的参数池（zero_grad_sparse / step_sparse 只激活领域）
    RoutingStrategy,           // 路由策略枚举（TopK / Threshold）
    RoutingResult,             // 路由结果（mask + active_indices + normalized_weights）
    DomainRouter,              // 稀疏路由主入口（route 决策 active_ids）
    build_param,               // 统一参数构造（自动命名 domain_{did}.{module}.{param}）
};

// ===== oyta-expert：专家模块 crate =====
pub use oyta_expert::{
    DomainExpert,              // 专家统一 trait（domain_id + forward + parameters + train/eval）
    Linear,                    // 全连接层（matmul + add）
    LayerNorm,                 // LayerNorm 归一化
    RMSNorm,                   // RMSNorm 归一化
    BatchNorm1d,               // BatchNorm1d（训练态/推理态切换 + running_mean/var + 缓存）
    Conv1d,                    // 一维卷积（纯 Vec 实现 + forward 缓存）
    Conv1dConfig,              // Conv1d 配置（in_channels / out_channels / kernel_size / stride / padding）
    MaxPool1d,                 // 最大池化（写 argmax 缓存）
    AvgPool1d,                 // 平均池化
    PoolConfig,                // 池化层配置
    set_training_mode,         // 全局训练态开关（供 Dropout / BN 使用）
    is_training_mode,          // 查询当前训练态
    apply_dropout,             // 带训练态判断的 Dropout（推理态直接 clone）
    MLPExpert,                 // MLP 专家（Linear → LN → GELU → Linear）
    CNN1DExpert,               // CNN1D 专家（Conv1d → BN → ReLU → Pool → Linear）
    CNN1DConfig,               // CNN1D 专家配置
    RNNExpert,                 // RNN 专家（骨架实现）
    TransformerExpert,         // Transformer 专家（LN → QKV → FlashAttn → o_proj → 残差 → FFN → proj）
    ExpertExecutor,             // 框架层稀疏专家执行器（continue 跳过未激活专家）
    BatchNormCache,            // BN forward 缓存（xhat + inv_std）
    Conv1dCache,               // Conv1d forward 缓存
    PoolCache,                 // Pool forward 缓存（argmax）
};

// ===== oyta-fusion：特征融合 + TaskHead crate =====
pub use oyta_fusion::{
    FusionStrategy,            // 融合策略枚举（WeightedSum / ConcatLinear / CrossAttention）
    FusionStrategyTrait,       // 融合策略统一 trait（fuse + fused_dim + parameters）
    TaskHeadBase,              // 任务头统一 trait（forward + output_dim + parameters）
    FeatureFusion,             // 策略分发主类（内部持有三选一实现）
    WeightedSumFusion,         // 加权求和融合（无可训练参数）
    ConcatLinearFusion,        // Concat + Linear 投影融合
    CrossAttentionFusion,      // FlashAttention + gating bias 注入融合
    TaskHead,                  // 任务头枚举（Classification / Regression）
    ClassificationHead,        // 分类任务头
    RegressionHead,            // 回归任务头
    concat_tensors,            // 任意维度 Tensor 拼接工具（纯 Vec 实现）
};

// ===== oyta-autograd：自动微分引擎 crate =====
pub use oyta_autograd::{
    Node,                      // 计算图节点（含 domain_id 稀疏核心字段 + requires_grad）
    NodeId,                    // 节点 ID 别名（u64）
    Op,                        // 算子枚举 17+ 变体（含 Conv1dMasked / BatchNorm1dMasked / Conv1dWithSequenceMask / Router 等）
    compute_backward_op,       // 计算单个 Op 的 backward 梯度（内部路由到对应实现）
    sparse_backward,           // 三层 domain_id 过滤的稀疏反向传播（核心：未激活领域子图 continue 跳过）
    AutogradEngine,            // 计算图引擎（register_node / register_leaf / backward / clear / training_mode）
};

// ===== oyta-loss：损失函数 crate =====
pub use oyta_loss::{
    Reduction,                 // 归约枚举（Mean / Sum / None）
    LossFn,                    // 损失函数统一 trait（compute + grad + item + reduction）
    CrossEntropyLoss,          // 交叉熵损失（label_smoothing 可选）
    MSELoss,                   // 均方误差损失
    MultiDomainLoss,           // 多领域联合损失（主损失 + 按 active_ids 过滤的辅助损失）
    log_softmax_stable,        // 数值稳定的 log_softmax 工具
    softmax_stable,            // 数值稳定的 softmax 工具（防极端值溢出）
};

// ===== oyta-optim：优化器 + Scheduler crate =====
pub use oyta_optim::{
    DomainState,               // 单个领域的独立优化状态（m / v / t）
    Optimizer,                 // 优化器统一 trait（zero_grad_sparse / step_sparse / clip_grad_norm_sparse）
    Scheduler,                 // 学习率调度器 trait（step / get_lr / set_lr / step_count）
    SparseAdam,                // 按领域独立维护 m/v/t 的稀疏 Adam（含 AdamW 解耦 weight decay）
    SparseSGD,                 // 稀疏 SGD（Momentum + Nesterov 可选）
    WarmupCosineAnneal,        // Warmup + CosineAnneal 组合调度器
    CosineAnneal,              // 纯 CosineAnnealing 调度器
    StepLR,                    // 固定步长衰减调度器
};

// ===== oyta-trainer：端到端训练 crate =====
pub use oyta_trainer::{
    SparsityConfig,            // 稀疏配置（enable_expert_sparsity / enable_channel_sparsity / force_all_experts_active debug 开关）
    TrainerConfig,             // 训练配置（max_epochs / gradient_accumulation_steps / max_grad_norm / seed）
    DataLoader,                // 数据加载器 trait（len / iter）
    Callback,                  // 回调 trait（on_step_end / on_epoch_end）
    SparseTrainer,             // 端到端稀疏训练主循环（11 步时序完整覆盖）
    SimpleDataLoader,          // 简易数据加载器（push / from_vec）
    LogCallback,               // 日志回调（每 N 步打印 loss + active_ids）
    SaveCallback,              // 保存回调（每 N epoch 提示 checkpoint）
    set_seed,                  // 设置全局随机种子（可复现性）
    get_seed,                  // 查询当前种子
    get_random_u64,            // 获取下一个随机 u64（seed 控制）
};

// ===== oyta-serialize：safetensors v0 格式序列化 crate =====
pub use oyta_serialize::{
    SafetensorsHeader,         // safetensors JSON header 结构定义
    TensorEntry,               // 单个 Tensor 在 header 中的 entry（dtype + shape + data_offsets）
    save_sparse_safetensors,   // 保存稀疏参数到 safetensors v0 文件（active_only 过滤可选）
    load_sparse_safetensors,   // 从 safetensors v0 文件加载（严格 shape 校验 + allow_partial 可选）
    build_param_key,           // 统一 key 命名：domain_{did}.{module}.{param}
    parse_param_key,           // key 反向解析 → (did, module, param)
};

// ===== prelude：一行导入全部（用户首选入口）=====
pub mod prelude {
    // 底层算子引擎
    pub use oyta_tensor::{Tensor, Cpu};
    // oyta-content
    pub use oyta_content::{
        Domain, RawInput, ContentAnalysis, Evidence,
        ContentAnalyzer, LabeledContent, FeatureEncoder, PaddingMask,
        RuleBasedAnalyzer, TFIDFAnalyzer, HybridAnalyzer,
    };
    // oyta-router
    pub use oyta_router::{
        SparseMaskSet, DomainSparseMask, MaskLevel, OpType,
        Parameter, SparseParameterPool,
        RoutingStrategy, RoutingResult, DomainRouter, build_param,
    };
    // oyta-expert
    pub use oyta_expert::{
        DomainExpert, Linear, LayerNorm, RMSNorm,
        BatchNorm1d, Conv1d, Conv1dConfig, MaxPool1d, AvgPool1d, PoolConfig,
        set_training_mode, is_training_mode, apply_dropout,
        MLPExpert, CNN1DExpert, CNN1DConfig, RNNExpert, TransformerExpert, ExpertExecutor,
        BatchNormCache, Conv1dCache, PoolCache,
    };
    // oyta-fusion
    pub use oyta_fusion::{
        FusionStrategy, FusionStrategyTrait, TaskHeadBase,
        FeatureFusion, WeightedSumFusion, ConcatLinearFusion, CrossAttentionFusion,
        TaskHead, ClassificationHead, RegressionHead, concat_tensors,
    };
    // oyta-autograd
    pub use oyta_autograd::{
        Node, NodeId, Op, compute_backward_op, sparse_backward, AutogradEngine,
    };
    // oyta-loss
    pub use oyta_loss::{
        Reduction, LossFn, CrossEntropyLoss, MSELoss, MultiDomainLoss,
        log_softmax_stable, softmax_stable,
    };
    // oyta-optim
    pub use oyta_optim::{
        DomainState, Optimizer, Scheduler,
        SparseAdam, SparseSGD, WarmupCosineAnneal, CosineAnneal, StepLR,
    };
    // oyta-trainer
    pub use oyta_trainer::{
        SparsityConfig, TrainerConfig, DataLoader, Callback,
        SparseTrainer, SimpleDataLoader, LogCallback, SaveCallback,
        set_seed, get_seed, get_random_u64,
    };
    // oyta-serialize
    pub use oyta_serialize::{
        SafetensorsHeader, TensorEntry,
        save_sparse_safetensors, load_sparse_safetensors,
        build_param_key, parse_param_key,
    };
    // 标准库常用基础类型（框架用户高频使用）
    pub use std::collections::HashMap;
    pub use std::sync::{Arc, Mutex};
}
