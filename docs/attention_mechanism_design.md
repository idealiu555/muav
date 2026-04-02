# 注意力机制设计方案

> 状态说明（2026-04-03 更新）：本文档已按当前代码实现校准。Attention 机制现已在 MADDPG 与 MAPPO 中落地，MATD3/MASAC 仍为无 attention 分支。最后更新时间：2026-04-02 实现，2026-04-03 文档验证。

## 1. 问题背景

### 原始问题：UE 观测截断
- UAV 只能观测固定数量的 UE（如 20 个），但实际可能服务更多
- 截断导致观测与实际状态不一致，影响决策质量
- 统计数据显示单个 UAV 最多可覆盖 26 个 UE

### 解决方案：注意力机制
- 使用 Cross-Attention 处理可变长度的 UE 列表
- 通过 padding + mask 支持动态数量的 UE
- 注意力机制自动学习关注重要的 UE

## 2. 架构设计：Cross-Attention

### 设计理念
- **Q (Query)**: 来自 UAV 自身状态，表示"UAV 想要关注什么"
- **K (Key)**: 来自 UE 特征，表示"每个 UE 的关键信息"
- **V (Value)**: 来自 UE 特征，表示"每个 UE 的实际内容"

### 语义解释
UAV 作为决策主体，主动查询（Query）UE 列表，根据每个 UE 的关键特征（Key）计算注意力权重，然后加权聚合 UE 的价值信息（Value）。

## 3. Embedding 设计

### 为什么需要 Embedding

1. **维度对齐**
   - UE 原始特征只有 5 维，维度太低
   - 注意力机制需要 64-128 维来捕捉复杂关系
   - 点积注意力要求 Q, K 维度一致

2. **特征类型差异**
   - **位置特征**（连续值）：需要线性变换
   - **文件 ID**（离散值）：需要 Embedding 层学习文件相似性
   - **缓存命中**（二值）：需要独立处理
   - **缓存位图**（多维二值）：需要学习缓存模式

3. **学习能力**
   - Embedding 可以学习特征之间的非线性关系
   - 例如：学习热门文件与冷门文件的区别

### Embedding 模块

#### UEEmbedding (5 → 128)
```
输入: [batch, num_ues, 5]
  - pos (3): 相对位置（归一化）
  - file_id (1): 请求文件 ID（归一化到 [0,1)，需要反归一化）
  - cache_hit (1): 缓存命中标志

处理:
  - pos → Linear(3, 64) → LeakyReLU(0.01)  # 正交初始化
  - file_id → round() → Embedding(NUM_FILES, 32)  # 学习文件相似性
  - cache_hit → Linear(1, 32) → LeakyReLU(0.01)  # 正交初始化
  - concat → LayerNorm(128)

输出: [batch, num_ues, 128]

维度分配比例: pos : file : cache_hit = 4 : 2 : 2 (64:32:32)
```

#### NeighborEmbedding (26 → 64) - 混合方案
```
输入: [batch, num_neighbors, 26]
  - pos (3): 相对位置
  - cache (20): 原始缓存位图（让注意力机制学习）
  - immediate_help (1): 即时帮助能力（预处理特征）
  - complementarity (1): 缓存互补性（预处理特征）
  - is_active (1): 活跃标记

处理:
  - pos → Linear(3, 16) → LeakyReLU(0.01)  # 正交初始化
  - cache → Linear(20, 32) → LeakyReLU(0.01)  # 信息最丰富，分配最多维度
  - processed → Linear(3, 16) → LeakyReLU(0.01)  # immediate_help + complementarity + is_active
  - concat → LayerNorm(64)

输出: [batch, num_neighbors, 64]

设计理念：
  - 原始 cache bitmap 让注意力机制自由学习协作模式
  - 预处理特征 (immediate_help, complementarity, is_active) 编码领域知识，加速收敛
  - 类似 ResNet shortcut：预处理特征提供"捷径"，原始数据提供细节

维度分配比例: pos : cache : processed = 1 : 2 : 1 (16:32:16)
```

#### UAVEmbedding (24 → 64)
```
输入:
  - pos: [batch, 3] 归一化位置
  - cache: [batch, NUM_FILES] 缓存位图
  - active: [batch, 1] 活跃标记

处理:
  - pos → Linear(3, 16) → LeakyReLU(0.01)  # 正交初始化
  - cache → Linear(20, 32) → LeakyReLU(0.01)  # 正交初始化
  - active → Linear(1, 16) → LeakyReLU(0.01)  # 正交初始化
  - concat → LayerNorm(64)

输出: [batch, 64]

维度分配比例: pos : cache : active = 1 : 2 : 1 (16:32:16)
```

## 4. Cross-Attention 实现

### 多头注意力
```
Q (Query) - 来自 UAV 状态:
  输入: UAV embedding [batch, 64]
  处理: Linear(64, kv_dim) + 正交初始化
  输出: [batch, 1, kv_dim]

K (Key) - 来自 UE/Neighbor 特征:
  输入: embeddings [batch, seq_len, kv_dim]
  处理: Linear(kv_dim, kv_dim) + 正交初始化
  输出: [batch, seq_len, kv_dim]

V (Value) - 来自 UE/Neighbor 特征:
  输入: embeddings [batch, seq_len, kv_dim]
  处理: Linear(kv_dim, kv_dim) + 正交初始化
  输出: [batch, seq_len, kv_dim]

注意力计算:
  Q, K, V = reshape to multi-head format [batch, num_heads, seq_len, head_dim]
  scores = Q @ K^T / sqrt(head_dim)  # scale factor
  scores = masked_fill(scores, mask==0, -inf)
  weights = softmax(scores, dim=-1)
  weights = nan_to_num(weights, nan=0.0)  # 处理全 mask 情况
  output = weights @ V
  output = reshape [batch, 1, kv_dim] → [batch, kv_dim]
  output = out_proj(output)  # 最终投影
```

### 维度配置
| 注意力类型 | query_dim | kv_dim | num_heads | head_dim |
|-----------|-----------|--------|-----------|----------|
| UE Attention | 64 | 128 | 2 | 64 |
| Neighbor Attention | 64 | 64 | 2 | 32 |

### Mask 机制
注意力机制广泛使用了 Mask 策略处理动态实体数量和环境失效：
1. **实体动态数量 Mask（UE/邻居数量）**：
   - 观测中包含 `neighbor_count` 和 `ue_count` 字段
   - 动态生成 mask：`mask[i] = 1 if i < count else 0`
    - 在 `CrossAttention` 中通过 `masked_fill(scores, mask==0, -inf)` 屏蔽 padding 位置，随后 `softmax` + `nan_to_num` 保证全 mask 行数值稳定。

2. **活跃智能体 Mask（Active Mask）**：
   - 在 Critic 的全局 Q 值评估聚合阶段（`AgentPoolingAttention`）不仅接受所有 Agent 状态，同时也接受 `active_mask` 指定的存活状态。
   - `key_padding_mask` / token 屏蔽同时作用以避免失效 / 获取的填充代理干扰全局决策特征。
   - 对失效实体组成的 batch 行引入全 inactive 兜底返回处理防止梯度和数值异常。

## 5. 完整编码器架构

### AttentionEncoder
```
输入: obs [batch, 380]

1. 解析观测 → 结构化数据
   - uav_pos: [batch, 3]
   - uav_cache: [batch, 20]
   - uav_active: [batch, 1]
   - neighbor_features: [batch, 4, 26]  # 混合特征
   - neighbor_count: [batch]
   - ue_features: [batch, 50, 5]
   - ue_count: [batch]

2. UAV Embedding → [batch, 64]

3. UE Embedding + CrossAttention
   - UE features → UEEmbedding → [batch, 50, 128]
   - CrossAttention(UAV_emb, UE_emb, mask) → [batch, 128]

4. Neighbor Embedding + CrossAttention
   - Neighbor features → NeighborEmbedding → [batch, 4, 64]
   - CrossAttention(UAV_emb, Neighbor_emb, mask) → [batch, 64]

5. 拼接 → [batch, 256]

输出: encoded [batch, 256]
```

## 6. 参数配置

### 当前配置（最佳实践）
| 参数 | 值 | 说明 |
|------|-----|------|
| USE_ATTENTION | False | **默认关闭**，需显式设置为 True 启用 |
| MAX_ASSOCIATED_UES | 50 | 最多观测 UE 数（覆盖 >99.9% 情况）|
| MAX_UAV_NEIGHBORS | 4 | 最多观测邻居数 |
| OWN_STATE_DIM | 24 | 自身状态维度 (pos(3) + cache(20) + active(1)) |
| NEIGHBOR_STATE_DIM | 26 | 邻居特征维度（混合方案）|
| UE_STATE_DIM | 5 | UE 特征维度 |
| OBS_DIM_SINGLE | 380 | 单 UAV 观测维度 |
| ATTENTION_EMBED_DIM | 128 | UE 注意力 embedding 维度 |
| ATTENTION_UAV_EMBED_DIM | 64 | UAV embedding 维度 |
| ATTENTION_NEIGHBOR_DIM | 64 | Neighbor 注意力维度 |
| ATTENTION_NUM_HEADS | 2 | 多头注意力头数（UE/Neighbor attention）|

### AgentPoolingAttention 参数（代码实际使用）
| 参数 | 值 | 说明 |
|------|-----|------|
| encoder_dim | 256 | 输入编码维度 |
| action_embed_dim | 64 | 动作嵌入维度 |
| num_heads | 4 | Self-Attention 头数（MADDPG 实例化时指定为 4，模块默认值为 8） |
| head_dim | 80 | 每个头的维度 (320/4) |
| FFN expansion | 2x | FFN 扩展比例 |

### 维度分析
| 组件 | heads | head_dim | 评价 |
|------|-------|----------|------|
| UE attention | 2 | 64 | 优秀（最佳实践 32-64）|
| Neighbor attention | 2 | 32 | 良好（最佳实践 32-64）|

### 输入/输出比
- 输入维度: 380
- 输出维度: 256
- 压缩比: 1.48（有效压缩，提取关键信息）

## 7. 观测结构

### 观测格式 (380 维)
```
[uav_pos(3), uav_cache(20), uav_is_active(1),
 neighbor_features(4×26=104), neighbor_count(1),
 ue_features(50×5=250), ue_count(1)]
```

### 邻居特征结构 (26 维/邻居)
```
[pos(3), cache_bitmap(20), immediate_help(1), complementarity(1), is_active(1)]
```

### UE 特征结构 (5 维)
```
[pos(3), file_id(1), cache_hit(1)]
```

## 8. 网络集成

### MADDPG ActorNetworkWithAttention
```
obs [batch, 380]
  → AttentionEncoder → [batch, 256]
  → Linear(256, 768) → LayerNorm → SiLU()  # 正交初始化
  → Linear(768, 768) → LayerNorm → SiLU()  # 正交初始化
  → Linear(768, 5) → Tanh  # std=0.01 正交初始化（输出层）
  → action [batch, 5]
```

**特点**：
- 独立编码器：每个 Actor 有自己的 AttentionEncoder（MADDPG 为多 Actor 结构）
- 输出层小 std 初始化：避免初始动作过于激进

### MADDPG CriticNetworkWithAttention（共享编码器架构）
```
joint_obs [batch, N×380]
  → 共享 AttentionEncoder（一次前向）→ [batch, N, 256]
  → AgentPoolingAttention（与 joint_action 聚合）→ [batch, 256]
  → Linear(256, 768) → LayerNorm → SiLU()
  → Linear(768, 768) → LayerNorm → SiLU() + Residual
  → Linear(768, 768) → LayerNorm → SiLU() + Residual
  → Linear(768, 1)
  → Q-value [batch, 1]
```

**特点**：
- 共享编码器：所有 Critic 共用同一个 AttentionEncoder
- 置换不变聚合：AgentPoolingAttention 输出维度与 N 无关
- 残差连接：MLP 层有残差连接，改善梯度流
- 参数分离：mlp_parameters() 返回 MLP + AgentPooling 参数（不含编码器）

### AgentPoolingAttention 架构
```
agent_encodings [batch, N, 256] + agent_actions [batch, N, 5] + active_mask [batch, N]
  → action_embed(actions) → [batch, N, 64]  # Linear(5, 64) + LayerNorm + LeakyReLU
  → concat(encodings, action_embed) → [batch, N, 320]  # 256 + 64
  → token_mask (通过 active_mask) 屏蔽失效 Agent 特征
  → Self-Attention(num_heads=4, head_dim=80) (附带 key_padding_mask) → [batch, N, 320]
  → 再次 token_mask + Residual → [batch, N, 320]
  → FFN(320 → 640 → 320) + token_mask + Residual → [batch, N, 320]
  → Masked Mean Pooling (仅平均 valid active 代理) → [batch, 320]
  → output_proj(320 → 256) → [batch, 256]
```

**参数配置**：
- encoder_dim: 256 (来自 AttentionEncoder 输出)
- action_embed_dim: 64
- num_heads: 4（在 `CriticNetworkWithAttention` 构造时固定传入）
- head_dim: 320 / 4 = 80 (符合最佳实践 64-128)
- FFN expansion: 2x (320 → 640 → 320)

**优势**：
- 共享编码器：减少参数量，提高训练效率
- AgentPooling：置换不变性，输出维度与 N 无关
- Self-Attention：agents 之间信息交互
- 残差连接：Self-Attention 和 FFN 都有残差，改善梯度流

### MAPPO AttentionActorNetwork
```
obs [batch, 380]
  → AttentionEncoder → [batch, 256]
  → _GaussianPolicyHead:
      Linear(256, 768) → LayerNorm → SiLU()
      Linear(768, 768) → LayerNorm → SiLU()
      mean_head(768, action_dim)
      global log_std parameter
  → Normal(mean, std)
  → tanh-squash（在 MAPPO 主逻辑中执行并做 Jacobian 校正）
```

**特点**：
- 共享策略网络：MAPPO 使用单一共享 actor（不是每个 agent 一个 actor）。
- actor/critic 编码器解耦：MAPPO attention 分支中 actor 与 critic 各自维护独立 AttentionEncoder。

### MAPPO AttentionCriticNetwork
```
joint_obs [batch, N, 380]
  → AttentionEncoder（批量编码）→ [batch, N, 256]
  → AgentPoolingValue(active_mask) → [batch, 256]
  → concat(one_hot(agent_id)) → [batch, 256 + N]
  → _AgentConditionedValueHead:
      Linear(256+N, 768) → LayerNorm → SiLU()
      Linear(768, 768) → LayerNorm → SiLU() + Residual
      Linear(768, 768) → LayerNorm → SiLU() + Residual
      Linear(768, 1)
  → V(joint_obs, agent_id)
```

**特点**：
- value 聚合不引入动作：使用 `AgentPoolingValue`（action-free）进行 centralized value 上下文构建。
- 支持 `joint_obs_index`：与 rollout flatten 后的 `(time, agent)` 样本对齐。

## 9. AgentPoolingValue 模块说明

### 设计与实现
`AgentPoolingValue` 是 action-free 的 agent-level 聚合模块，用于 MAPPO 和其他 on-policy 算法中的 centralized value 估计。

**与 `AgentPoolingAttention` 的区别**：
- `AgentPoolingAttention`：接收共享编码+动作，用于 MADDPG 的 action-conditioned Q-value
- `AgentPoolingValue`：仅接收共享编码，用于 MAPPO 的 action-free value 估计

### 架构设计
```
agent_encodings [batch, N, encoder_dim] + active_mask [batch, N]
  → Self-Attention(num_heads=8, multi-head=True)
  → token_mask（通过 active_mask）屏蔽失效 agent
  → Residual + LayerNorm
  → FFN(encoder_dim → encoder_dim*2 → encoder_dim)
  → token_mask 再次应用
  → Residual + LayerNorm
  → Masked Mean Pooling（仅平均有效 agent）→ [batch, encoder_dim]
  → output_proj [batch, encoder_dim] → [batch, encoder_dim]
```

### 关键设计原则
1. **Permutation-Invariant**：被 mask 的代理对输出无影响，对代理顺序不敏感
2. **All-Inactive 处理**：全 inactive batch 行返回零向量（通过 `new_zeros` + `valid_rows` mask）
3. **Token-Level Masking**：每个无效 agent 的特征被逐元素置零，而非整体跳过
4. **Residual 连接**：Self-Attention 和 FFN 都有残差，改善深层网络的梯度流

### 实现细节（来自代码）
```python
class AgentPoolingValue(nn.Module):
    def __init__(self, encoder_dim: int = 256, num_heads: int = 8) -> None:
        # Self-Attention 层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=encoder_dim, num_heads=num_heads, batch_first=True
        )
        
        # LayerNorm 和 FFN
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.LeakyReLU(0.01),
            nn.Linear(encoder_dim * 2, encoder_dim),
        )
        self.ffn_norm = nn.LayerNorm(encoder_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(encoder_dim, encoder_dim)
    
    def forward(self, agent_encodings: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        # agent_encodings: [batch, N, encoder_dim]
        # active_mask: [batch, N] (dtype=torch.bool)
        # 返回: [batch, encoder_dim]
        # 全 inactive batch 行返回零向量
```

### Masked Mean Pooling
```python
# 仅平均有效 agent 的特征，返回固定维度输出
denominator = active_mask.sum(dim=1).clamp_min(1.0)  # [batch]
pooled = (masked_features * active_mask.unsqueeze(-1)).sum(dim=1) / denominator
```

## 10. 关键实现细节

### Bug 修复：File ID 反归一化 + round()
```python
# 错误实现（所有 file_id 都被映射到 0）
file_id = ue_features[:, :, 3].long()  # 0.5.long() = 0 ← 错误!

# 改进实现（使用 round() 避免浮点精度问题）
file_id = (ue_features[:, :, 3] * config.NUM_FILES).round().long().clamp(0, config.NUM_FILES - 1)
# 0.4999999 * 20 = 9.999998 → round() = 10 ← 正确!
```

### 全 Mask 处理
```python
# 当所有位置都是 padding 时，softmax 会产生 nan
attn_weights = F.softmax(attn_scores, dim=-1)
attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # 将 nan 替换为 0
```

### 观测格式（统一包含 count 字段）
```python
# env.py: 始终包含 count 字段（attention 和 non-attention 模式均可用）
obs = np.concatenate([..., neighbor_count, ..., ue_count])
```

### 共享编码器梯度管理（MADDPG）
```python
# 梯度累积：所有 agent 的 critic 梯度在共享编码器上累积
retain_graph = config.USE_ATTENTION and not is_last_agent
critic_loss.backward(retain_graph=retain_graph)

# 梯度缩放：按活跃 agent 数量缩放，保持梯度量级一致
encoder_grad_scale = max(1, updated_agents)
for param in self.shared_encoder.parameters():
    if param.grad is not None:
        param.grad.div_(encoder_grad_scale)

# 梯度裁剪：共享编码器单独裁剪
torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), config.MAX_GRAD_NORM)
```

### Critic 参数冻结（Actor 更新时）
```python
# Actor 更新时冻结 Critic 参数，防止 Actor 影响 Critic 表示学习
for param in self.critics[agent_idx].parameters():
    param.requires_grad_(False)
actor_q = self.critics[agent_idx](obs_flat, pred_actions_flat, actor_joint_encoded)
actor_loss.backward()
for param in self.critics[agent_idx].parameters():
    param.requires_grad_(True)
```

### Target 网络更新（Soft Update）
```python
# 共享编码器：只更新一次（所有 Critic 共享）
soft_update(self.target_shared_encoder, self.shared_encoder, config.UPDATE_FACTOR)

# Critic MLP：各 agent 分别更新（仅 MLP 参数，不包括共享编码器）
for agent_idx in range(self.num_agents):
    with torch.no_grad():
        for target_param, param in zip(self.target_critics[agent_idx].mlp_parameters(),
                                       self.critics[agent_idx].mlp_parameters()):
            target_param.copy_(config.UPDATE_FACTOR * param + (1.0 - config.UPDATE_FACTOR) * target_param)
```

## 11. 使用方式

### 配置开关
```python
# config.py
USE_ATTENTION: bool = False  # 默认关闭，需显式设置为 True 启用
```

### MADDPG 中的集成
```python
# maddpg.py
if config.USE_ATTENTION:
    # 创建真正的单一共享编码器（所有 Critic 共用）
    self.shared_encoder = AttentionEncoder().to(device)
    self.target_shared_encoder = AttentionEncoder().to(device)
    
    # Actor 使用独立编码器（每个 Actor 有自己的 AttentionEncoder）
    self.actors = [ActorNetworkWithAttention(obs_dim, action_dim) for _ in range(num_agents)]
    
    # Critic 使用共享编码器（传入 shared_encoder 引用）
    self.critics = [CriticNetworkWithAttention(..., self.shared_encoder) for _ in range(num_agents)]
    
    # 优化器结构
    # 1. 共享编码器：单独的优化器（所有 Critic 共享）
    self.shared_encoder_optimizer = AdamW(self.shared_encoder.parameters(), lr=CRITIC_LR)
    # 2. Critic MLP：每个 Critic 有独立优化器（仅包含 MLP + AgentPooling 参数）
    self.critic_optimizers = [AdamW(critic.mlp_parameters(), lr=CRITIC_LR) for critic in self.critics]
else:
    self.actors = [ActorNetwork(obs_dim, action_dim) for _ in range(num_agents)]
    self.critics = [CriticNetwork(num_agents, obs_dim, action_dim) for _ in range(num_agents)]
```

### 参数分离策略
| 组件 | 优化器 | 说明 |
|------|--------|------|
| Actor 编码器 | Actor 优化器 | 每个 Actor 独立，包含完整网络参数 |
| Critic 共享编码器 | shared_encoder_optimizer | 所有 Critic 共享一个优化器 |
| Critic MLP + AgentPooling | critic_optimizers[i] | 每个 Critic 独立，仅 MLP 参数 |

**梯度流**：
- Critic 更新：梯度在共享编码器上累积 → 最后统一更新
- Actor 更新：独立更新，不影响共享编码器

### MAPPO 中的集成
```python
# mappo.py
actor_cls = AttentionActorNetwork if config.USE_ATTENTION else ActorNetwork
critic_cls = AttentionCriticNetwork if config.USE_ATTENTION else CriticNetwork

self.actors = actor_cls(obs_dim, action_dim).to(device)
self.critics = critic_cls(num_agents, obs_dim).to(device)

# Critic 统一输入契约（attention / non-attention 一致）
values = self.critics(
  joint_obs_batch,
  agent_index_batch,
  joint_active_mask_batch,
  sample_index=joint_obs_index_batch,
)
```

**说明**：MAPPO 已去除 `global_state` 训练路径，统一使用 `joint_obs + agent_index + active_mask`。训练时通过 `joint_active_mask + joint_obs_index` 处理去重后的 joint context；在线动作采样时使用单 joint context（`[1, N, obs_dim]`）配合多 agent 查询。

```python
# get_action_and_value() 中的在线 value 评估（当前实现）
agent_index = torch.arange(self.num_agents, device=self.device, dtype=torch.long)
critic_joint_obs = obs_tensor.unsqueeze(0)          # [1, N, obs_dim]
critic_active_mask = active_mask.unsqueeze(0).bool()  # [1, N]
values = self.critics(critic_joint_obs, agent_index, critic_active_mask)
```

## 11. 性能考虑

### 计算复杂度
- UE attention: O(50 × 128) per UAV
- Neighbor attention: O(4 × 64) per UAV
- AgentPooling: O(N² × 320) for N agents
- 总体增加约 25% 计算量，但信息完整性大幅提升

### 参数效率
- 共享编码器：Critic 共享一个 AttentionEncoder，参数量 O(1) 而非 O(N)（MADDPG attention 分支）
- Actor 编码器：MADDPG 中每个 Actor 有独立编码器；MAPPO 中为单共享 actor 编码器

### 训练效率优化
```python
# 1. 预计算编码（减少重复前向传播）
joint_encoded = self.critics[0].encode_observations(obs_flat)
target_joint_encoded = self.target_critics[0].encode_observations(next_obs_flat)

# 2. 预计算动作（减少 Actor 前向传播从 O(N²) 到 O(2N)）
with torch.no_grad():
    all_actions_detached = [self.actors[i](obs_tensor[:, i, :]) for i in range(num_agents)]
    next_actions = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(num_agents)]

# 3. 批量化编码（一次前向传播处理所有 agent）
reshaped = joint_obs.view(batch_size * num_agents, obs_dim)
encoded = self.encoder(reshaped)  # [batch * N, encoder_dim]
```

### 内存优化
- Actor 更新时 detach critic 编码：`actor_joint_encoded = joint_encoded.detach()`
- 预计算动作使用 `torch.no_grad()` 避免存储中间梯度

## 12. 设计决策：混合邻居特征

### 为什么采用混合方案？

| 方案 | 维度 | 信息量 | 学习难度 |
|------|------|--------|----------|
| 仅预处理特征 | 6 | 部分丢失 | 低 |
| 仅原始 cache | 24 | 完整 | 高 |
| **混合方案** | 26 | 完整+先验 | 中 |

### 混合方案的优势
1. **原始 cache bitmap**：让注意力机制自由学习协作模式
2. **预处理特征**：编码领域知识（即时帮助、互补性、活跃状态），加速收敛
3. **类似 ResNet shortcut**：预处理特征提供"捷径"，原始数据提供细节

## 13. 与其他算法的兼容性

**当前实现状态（以代码为准）**：

| 算法 | Attention 支持 | 说明 |
|------|----------------|------|
| MADDPG | ✅ 完整支持 | Actor + Critic 均有 Attention 版本 |
| MAPPO | ✅ 完整支持 | 支持 AttentionActorNetwork + AttentionCriticNetwork + AgentPoolingValue |
| MATD3 | ❌ 不支持 | 使用独立 MLP Actor/双 Critic（无 Attention 编码器） |
| MASAC | ❌ 不支持 | 使用独立 MLP Actor/双 Critic（无 Attention 编码器） |

补充：`MeanPoolingEncoder` 当前用于 MADDPG 与 MAPPO 的无注意力分支，用来减少 padding 对输入语义的污染。

## 14. 未来改进方向

1. **Actor 编码器共享**：使所有 Actor 共享一个编码器，减少参数量
2. **其他算法支持**：为 MATD3、MASAC 实现 Attention 版本
3. **位置编码**：为 UE 添加相对位置编码，增强空间感知
4. **多层注意力**：堆叠多层 attention 捕捉更复杂的关系

## 15. 扩展性设计

### 动态 Agent 数量支持
```python
# CriticNetworkWithAttention.encode_observations 支持动态 num_agents
def encode_observations(self, joint_obs: torch.Tensor, num_agents: int | None = None):
    if num_agents is None:
        num_agents = self.num_agents
    # 处理任意数量的 agent
    batch_size = joint_obs.shape[0]
    reshaped = joint_obs.view(batch_size, num_agents, self.obs_dim)
    ...
```

### 应用场景
- 训练时 UAV 故障：活跃 UAV 数量动态变化
- 测试时扩展：使用不同数量的 UAV
- 模型迁移：将训练好的模型应用到不同规模的场景

**关键设计**：
- AgentPoolingAttention 输出维度与 agent 数量无关
- encode_observations 通过参数动态调整
- save/load 时编码器单独存储，便于迁移
