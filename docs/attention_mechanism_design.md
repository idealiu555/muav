# 注意力机制设计方案

> 状态说明（2026-06-06 更新）：MAPPO 已移除 attention 分支，只保留标准 MLP actor 和 centralized MLP critic。当前 attention 模块仅服务于 MASAC 的可选分支；MADDPG、MATD3、MAPPO 均不构造 attention encoder。

## 1. 当前结论

- `MODEL="mappo"` 不提供 attention actor 或 attention critic。
- `MODEL="masac"` 使用独立的 attention 控制：`MASAC_ATTENTION_ACTOR` 控制 actor 是否使用 attention encoder，`MASAC_CRITIC_MODE` 控制 critic 类型（`"mlp"` / `"local_attention"` / `"agent_self_attention"`）。
- `MODEL="maddpg"` 和 `MODEL="matd3"` 没有 attention 分支。
- `marl_models/attention.py` 当前保留给 MASAC attention 分支复用。

## 2. 为什么还保留 AttentionEncoder

环境观测中包含两类可变长度实体：

- 邻居 UAV 列表：最多 `MAX_UAV_NEIGHBORS`
- 关联 UE 列表：最多 `MAX_ASSOCIATED_UES`

环境为了保持张量形状固定，会对这两部分做 padding，并在观测中附带：

- `neighbor_count`
- `ue_count`

`AttentionEncoder` 会用这两个计数字段构造 mask，从而忽略 padding 项、仅聚合真实实体，并在全零/空列表情况下保持数值稳定。这个能力目前用于 MASAC 的 attention actor、local-attention critic 和 agent-self-attention critic。

## 3. MASAC Attention 架构

### 3.1 Actor

当 `MASAC_ATTENTION_ACTOR=True`：

```text
obs_i
  -> AttentionEncoder
  -> encoded_obs_i
  -> Gaussian policy head
  -> action_i
```

当 `MASAC_ATTENTION_ACTOR=False`，actor 直接使用 raw observation 进入 MLP。

### 3.2 Local-Attention Critic

当 `MASAC_CRITIC_MODE="local_attention"`：

```text
obs_i for each agent
  -> shared AttentionEncoder
  -> flatten all encoded observations
  -> concat joint actions
  -> MLP critic
  -> per-agent Q values
```

这个模式只做每个 agent 局部观测内的实体聚合，不做 agent-level self-attention。

### 3.3 Agent-Self-Attention Critic

当 `MASAC_CRITIC_MODE="agent_self_attention"`：

```text
obs_i for each agent
  -> shared AttentionEncoder
  -> concat action_i and agent identity embedding
  -> masked agent-level self-attention
  -> shared Q head
  -> per-agent Q values
```

这个模式在局部实体编码之上显式建模 UAV 之间的依赖关系。

## 4. MAPPO 当前架构

MAPPO 当前只有无 attention 分支：

```text
actor:  obs_i -> MLP Gaussian policy head
critic: share_obs(flattened joint obs) -> centralized MLP value head -> per-agent values
```

相关代码：

- `marl_models/mappo/agents.py`
- `marl_models/mappo/mappo.py`

## 5. 测试建议

attention 相关回归测试应关注：

- `AttentionEncoder` 输出 shape 正确
- 全零 count 时无 NaN
- padding 区域的值不会泄漏到编码结果
- MASAC attention actor/critic 在 rollout 和 update 中保持正确接口
- MAPPO 不导出 attention actor/critic，也不依赖全局 attention 开关
