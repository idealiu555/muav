# MAPPO 说明

## 项目定位

这个项目把低空网络中的通信保障建模成多智能体的飞行轨迹与 3D 波束赋形联合优化问题。多架 UAV 在每个时隙同时决策三维位移和波束方向，并在系统速率、服务时延、能耗和公平性之间做权衡。

当前仓库中的 MAPPO 是无 attention 的标准基线实现：actor 使用单个 UAV 的局部观测输出随机连续动作，centralized critic 使用展平后的全体 UAV 观测输出 per-agent value。MAPPO 不再构造 `AttentionEncoder`，也不再提供 attention actor/critic 分支。

## 算法结构

```text
actor:
  obs_i -> MLP Gaussian policy head -> tanh-squashed action_i

critic:
  share_obs(flattened joint obs)
    -> centralized MLP value head
    -> per-agent values
```

实现位置：

- `marl_models/mappo/agents.py`
- `marl_models/mappo/mappo.py`
- `marl_models/mappo/rollout_buffer.py`

## 面试高频问答

### 问题1：这个项目的核心研究问题是什么，为什么要做联合优化？

核心问题是低空网络里 UAV 的移动控制和 3D 波束赋形强耦合。轨迹决定覆盖关系、链路距离和干扰结构；波束方向又直接影响链路增益和同频干扰。单独优化轨迹或波束都容易得到局部方案，所以项目把两者统一放进多智能体连续控制问题中，从系统速率、服务时延、能耗和公平性四个维度共同优化。

### 问题2：为什么选择 MAPPO 作为基线算法？

MAPPO 适合作为这个场景的 on-policy 多智能体基线。它采用集中训练、分散执行，actor 基于局部观测决策，critic 在训练时使用联合观测估计价值。相比确定性策略方法，PPO 类方法的策略更新有 clipping 约束，在高维连续动作和非平稳多智能体环境中通常更稳定。

### 问题3：状态空间和动作空间怎么定义？

每个 UAV 的局部观测包含自身状态、邻居 UAV 信息、当前服务 UE 信息，以及邻居数量和关联 UE 数量字段。当前 MAPPO 将这些固定长度观测直接输入 MLP。动作空间是连续动作：前三维控制 x、y、z 方向的归一化位移；启用波束控制时，后两维控制波束俯仰角和方位角。

### 问题4：centralized critic 怎么工作？

MAPPO 的 critic 接收所有 agent 观测拼接后的 `share_obs`，形状为 `[batch, num_agents * obs_dim]`，通过集中式 MLP 输出 `[batch, num_agents]` 的 value 向量。训练 minibatch 中再根据 `agent_index` 取出对应 agent 的 value，用于 value loss 和 advantage 计算。

### 问题5：实现里有哪些稳定性处理？

当前实现使用 tanh-squashed Gaussian policy 适配环境的有界动作契约，并在 log-prob 中加入 tanh Jacobian 修正。训练侧使用 PPO clipped surrogate、value clipping、GAE、ValueNorm、active mask、梯度裁剪、正交初始化和 LayerNorm。

### 问题6：奖励函数怎么避免模型只追求某一个指标？

奖励是多目标加权形式：总时延和总能耗作为惩罚项，系统速率和 Jain 公平性指数作为奖励项，并通过尺度归一化和对数压缩降低量纲差异。碰撞、危险接近和越界也有显式惩罚，避免策略为了短期吞吐做出不安全动作。
