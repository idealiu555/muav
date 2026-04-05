# 注意力机制设计方案

> 状态说明（2026-04-06 更新）：本文档已按当前仓库实现重新整理。**注意力机制现在只保留在 MAPPO 中**；MADDPG 已移除历史上的 attention / pooling 分支，恢复为标准 MLP actor + centralized MLP critic。

## 1. 当前结论

- `USE_ATTENTION` 只影响 `MODEL="mappo"`。
- `MODEL="maddpg"` 无论 `USE_ATTENTION=True/False`，都不会构造 attention encoder、shared encoder、mean pooling 或 agent pooling。
- `marl_models/attention.py` 当前只保留 MAPPO 仍在使用的基础模块：
  - `parse_observation`
  - `UAVEmbedding`
  - `UEEmbedding`
  - `NeighborEmbedding`
  - `CrossAttention`
  - `AttentionEncoder`

## 2. 为什么还保留 AttentionEncoder

环境观测中包含两类可变长度实体：

- 邻居 UAV 列表：最多 `MAX_UAV_NEIGHBORS`
- 关联 UE 列表：最多 `MAX_ASSOCIATED_UES`

环境为了保持张量形状固定，会对这两部分做 padding，并在观测尾部附带：

- `neighbor_count`
- `ue_count`

`AttentionEncoder` 会用这两个计数字段构造 mask，从而：

- 忽略 padding 项
- 仅聚合真实实体
- 在全零/空列表情况下保持数值稳定

## 3. 当前 MAPPO Attention 架构

### 3.1 Actor

输入：

- 单个 agent 的局部观测 `obs_i`

路径：

```text
obs_i
  -> AttentionEncoder
  -> encoded_obs_i
  -> GaussianPolicyHead
  -> Normal distribution
```

说明：

- `AttentionEncoder` 先把 flat observation 解析成 UAV / Neighbor / UE 三部分结构化特征。
- UAV 自身特征作为 query。
- Neighbor 和 UE 特征分别作为 key/value 做 cross-attention。
- 最终输出固定维度编码，再送入策略头。

### 3.2 Critic

输入：

- `share_obs`，形状为 `[batch, num_agents * obs_dim]`

路径：

```text
share_obs
  -> reshape([batch, N, obs_dim])
  -> per-agent AttentionEncoder
  -> agent_encodings [batch, N, 256]
  -> flatten(team_context) [batch, N*256]
  -> LayerNorm(team_context)
  -> broadcast 到每个 agent
  -> concat(team_context, e_i)
  -> shared scalar value head
  -> per-agent values [batch, N]
```

说明：

- 当前 critic 输出的是 `V(s, agent_i)`，不是单一标量 `V(s)`。
- 这让 attention MAPPO 可以对 agent-specific reward/penalty 建模。

## 4. AttentionEncoder 结构

`AttentionEncoder` 的核心结构如下：

```text
flat obs
  -> parse_observation
  -> UAVEmbedding            -> [batch, 64]
  -> UEEmbedding             -> [batch, max_ues, 128]
  -> NeighborEmbedding       -> [batch, max_neighbors, 64]
  -> UE CrossAttention       -> [batch, 128]
  -> Neighbor CrossAttention -> [batch, 64]
  -> concat                  -> [batch, 256]
```

默认维度：

- UAV embedding: `64`
- UE embedding: `128`
- Neighbor output: `64`
- Encoder output: `256`
- Attention heads: `2`

## 5. Mask 机制

### 5.1 邻居 Mask

由 `neighbor_count` 生成：

```text
neighbor_mask[i, j] = (j < neighbor_count[i])
```

### 5.2 UE Mask

由 `ue_count` 生成：

```text
ue_mask[i, j] = (j < ue_count[i])
```

### 5.3 数值稳定性

当前实现要求：

- 空邻居集合时不产生 NaN
- 空 UE 集合时不产生 NaN
- padding 槽位的值变化不影响编码结果

这也是当前测试覆盖的重点。

## 6. 与 MADDPG 的关系

当前 **没有** MADDPG attention 分支。MADDPG 的真实实现为：

```text
actor:  obs_i -> MLP -> action_i
critic: concat(joint_obs, joint_action) -> MLP -> Q_i
```

MADDPG 仍保留的工程逻辑：

- `active_mask`
- `bootstrap_mask`
- `AdamW`
- `LayerNorm`
- 正交初始化
- 梯度裁剪

但这些都不再依赖 attention 模块。

## 7. 代码位置

- Attention 基础模块：[marl_models/attention.py](/C:/Users/mateogic/Desktop/muav/muav/marl_models/attention.py)
- MAPPO actor/critic：[marl_models/mappo/agents.py](/C:/Users/mateogic/Desktop/muav/muav/marl_models/mappo/agents.py)
- MAPPO 主逻辑：[marl_models/mappo/mappo.py](/C:/Users/mateogic/Desktop/muav/muav/marl_models/mappo/mappo.py)
- MADDPG 标准实现：[marl_models/maddpg/agents.py](/C:/Users/mateogic/Desktop/muav/muav/marl_models/maddpg/agents.py)、[marl_models/maddpg/maddpg.py](/C:/Users/mateogic/Desktop/muav/muav/marl_models/maddpg/maddpg.py)

## 8. 测试建议

当前 attention 相关回归测试应关注：

- `AttentionEncoder` 输出 shape 正确
- 全零 count 时无 NaN
- padding 区域的值不会泄漏到编码结果
- MAPPO attention actor/critic 在 rollout 和 update 中保持正确接口
