# MASAC Agent Self-Attention Design and Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current MASAC attention critic with a configurable critic family that can run raw MLP, local entity attention, or agent-level self-attention without changing SAC target semantics.

**Architecture:** Keep MASAC as centralized training with decentralized execution. Actor attention remains a local-observation encoder only; critic attention is split into two explicit modes: local entity encoding only, and agent self-attention over UAV tokens. The new self-attention critic estimates per-agent `Q_i(o_1..o_N, a_1..a_N)` by building one token per UAV and applying masked agent-level self-attention before a shared per-agent Q head.

**Tech Stack:** Python, PyTorch, pytest

---

## 1. Diagnosis

The current `AttentionCriticNetwork` in `marl_models/masac/agents.py` is not agent-level self-attention. Its logic is:

```text
joint_obs -> reshape to [batch * num_agents, obs_dim]
obs_i -> AttentionEncoder -> h_i
concat([h_1, ..., h_N], joint_action_flat) -> MLP -> [Q_1, ..., Q_N]
```

This only uses attention inside each agent's local observation to aggregate UE and neighbor slots. It still asks a flat MLP to infer inter-agent relevance. That is the same failure mode described in `attn_marl.md` for concatenation-based centralized critics: all agents are flattened into one vector, and the model must implicitly learn which other agents matter for each `Q_i`.

The replacement must preserve MASAC semantics:

- Critic estimates action values, not state values. Every critic path must consume both observations and actions.
- Actor remains decentralized at execution time.
- Twin critics remain independent.
- Existing per-agent loss masking remains authoritative.
- Old local-attention critic remains available for ablation instead of being overwritten.

## 2. Configuration Model

Do not use one `USE_ATTENTION` flag to control all MASAC attention behavior. Actor local encoding and critic relation modeling are orthogonal changes.

Add MASAC-specific config:

```python
MASAC_ATTENTION_ACTOR: bool = False
MASAC_CRITIC_MODE: str = "mlp"  # options: "mlp", "local_attention", "agent_self_attention"
MASAC_AGENT_ID_DIM: int = 32
MASAC_AGENT_ATTENTION_DIM: int = 256
MASAC_AGENT_ATTENTION_HEADS: int = 4
MASAC_AGENT_ATTENTION_LAYERS: int = 1
MASAC_AGENT_ATTENTION_FFN_MULT: int = 4
```

Mode behavior:

```text
MASAC_ATTENTION_ACTOR=false, MASAC_CRITIC_MODE="mlp"
    ActorNetwork + CriticNetwork

MASAC_ATTENTION_ACTOR=true, MASAC_CRITIC_MODE="mlp"
    AttentionActorNetwork + CriticNetwork

MASAC_CRITIC_MODE="local_attention"
    current local AttentionEncoder critic: encode each o_i, flatten all h_i, concat actions, MLP -> [Q_i]

MASAC_CRITIC_MODE="agent_self_attention"
    new critic: encode each o_i, build per-agent tokens from h_i/a_i/id_i, masked self-attention, QHead -> [Q_i]
```

Keep `USE_ATTENTION` only as a legacy/global default if needed. MASAC model construction should prefer MASAC-specific flags so ablations are explicit.

## 3. Network Design

### 3.1 Actor

Actor must stay local-only:

```text
o_i -> ActorNetwork or AttentionActorNetwork -> pi_i(a_i | o_i)
```

It must not receive joint observations, joint actions, active masks, or agent-level self-attention context. This preserves CTDE.

### 3.2 Critic Modes

Keep three critic classes or clearly named equivalents:

```text
CriticNetwork
    Raw flattened MLP critic.

LocalAttentionCriticNetwork
    Rename the current AttentionCriticNetwork behavior.
    It reuses AttentionEncoder per agent but has no agent self-attention.

AgentSelfAttentionCriticNetwork
    New MASAC attention critic matching attn_marl.md's agent-level self-attention pattern.
```

Do not keep `AttentionCriticNetwork` as an alias for either attention critic. The name already referred to the old local entity encoder critic in prior code, so reusing it for the new agent self-attention critic makes history and checkpoint debugging ambiguous. Update imports and tests to use `LocalAttentionCriticNetwork` and `AgentSelfAttentionCriticNetwork` explicitly, then delete the misleading old name.

### 3.3 Local Entity Encoder

Reuse `AttentionEncoder` from `marl_models/attention.py`.

Within one critic instance, use one shared `AttentionEncoder` for all agents:

```text
obs: [batch, num_agents, obs_dim]
obs.reshape(batch * num_agents, obs_dim) -> encoder -> [batch * num_agents, enc_dim]
reshape -> [batch, num_agents, enc_dim]
```

This is batched, simple, and more efficient than a Python for-loop over agents. Twin critics must each own their own encoder instance; do not share encoder parameters between `critic_1` and `critic_2`.

### 3.4 Agent Token Builder

For each agent:

```text
h_i = AttentionEncoder(o_i)
e_i = agent_id_embedding(i)
t_i = token_projection(concat(h_i, a_i, e_i))
```

Do not concatenate a raw integer ID. Do not include `active_i` as a token feature. Activity is already present in the observation and should control attention visibility through masks, not duplicate as a weak scalar in dense token content.

Build ID embeddings in one vectorized operation on the same device as `obs`:

```python
agent_ids = torch.arange(self.num_agents, device=obs.device)
agent_id_emb = self.agent_id_embedding(agent_ids)
agent_id_emb = agent_id_emb.unsqueeze(0).expand(batch_size, -1, -1)
tokens = torch.cat([encoded_obs, actions, agent_id_emb], dim=-1)
```

Do not loop over agents to assemble tokens.

Input/output dimensions:

```text
token_input_dim = AttentionEncoder.output_dim + action_dim + MASAC_AGENT_ID_DIM
token_dim = MASAC_AGENT_ATTENTION_DIM
```

Validate:

```text
MASAC_AGENT_ATTENTION_DIM % MASAC_AGENT_ATTENTION_HEADS == 0
MASAC_AGENT_ID_DIM > 0
MASAC_AGENT_ATTENTION_LAYERS >= 1
```

### 3.5 Agent Self-Attention Block

Use pre-LayerNorm residual blocks:

```text
y = LayerNorm(x)
y = MultiheadAttention(y, y, y, key_padding_mask=inactive_key_mask)
x = x + y

z = LayerNorm(x)
z = FFN(z)
x = x + z
```

The FFN is:

```text
FFN(x) = Linear(dim * ffn_mult -> dim)(SiLU(Linear(dim -> dim * ffn_mult)(x)))
```

Use `SiLU` for consistency with the existing MASAC actor and critic MLPs.

`agent_mask` follows repo semantics: `1=valid/visible`, `0=inactive`. PyTorch `nn.MultiheadAttention` uses the opposite convention for `key_padding_mask`, where `True` means masked. Convert explicitly:

```python
key_padding_mask = agent_mask <= 0
```

`key_padding_mask` shape is `[batch, num_agents]`, with `True` for inactive agents when using PyTorch `nn.MultiheadAttention`.

Guard all-masked batch rows before calling attention. If one sample has every agent inactive, `MultiheadAttention` may produce NaNs because every attention logit is masked. A safe implementation should temporarily unmask all keys for those rows, run attention, then zero their contexts afterward:

```python
valid_query_mask = agent_mask > 0
key_padding_mask = ~valid_query_mask
all_inactive = key_padding_mask.all(dim=1)
safe_key_padding_mask = key_padding_mask.clone()
safe_key_padding_mask[all_inactive] = False
```

Use `safe_key_padding_mask` in `MultiheadAttention`.

The key padding mask prevents inactive agents from contributing as keys/values. It does not stop inactive agents from acting as queries, so after the final attention block:

```python
context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
context = torch.where(
    valid_query_mask.unsqueeze(-1),
    context,
    torch.zeros_like(context),
)
```

Do this before the Q head. The loss mask still remains in MASAC update; zeroing context is an additional containment measure for cleaner values and diagnostics. Do not use `context * mask` as the only NaN defense, because `NaN * 0` remains `NaN`.

### 3.6 Q Head

Use a shared per-agent Q head:

```text
q_input_i = concat(context_i, h_i, a_i, e_i)
q_i = QHead(q_input_i)
```

The direct `h_i` and `a_i` skip connection is intentional. SAC actor updates depend on the critic gradient with respect to the current action. Passing `a_i` only through token projection, self-attention, and normalization makes that gradient path unnecessarily indirect. Concatenating `a_i` again at the Q head gives the actor a clear local `dQ_i/da_i` path while preserving the agent-level context learned by self-attention.

After the Q head, set inactive agents' final Q values to zero with `torch.where(valid_query_mask, q_values, zeros)`. This keeps inactive outputs finite and inert even though the Q head receives skip-connected local features.

Return:

```text
[batch, num_agents]
```

The learned agent ID embedding gives the shared Q head agent-specific conditioning. This is the parameter-efficient equivalent of separate `QHead_i` layers from the reference architecture.

## 4. Forward Interfaces

### 4.1 Raw MLP Critic

Keep unchanged:

```python
forward(joint_obs_flat: torch.Tensor, joint_action_flat: torch.Tensor) -> torch.Tensor
```

Shapes:

```text
joint_obs_flat: [batch, num_agents * obs_dim]
joint_action_flat: [batch, num_agents * action_dim]
return: [batch, num_agents]
```

### 4.2 Local Attention Critic

Keep the old flattened interface if preserving existing tests is cheaper:

```python
forward(joint_obs_flat: torch.Tensor, joint_action_flat: torch.Tensor) -> torch.Tensor
```

This mode is only for ablation and compatibility with existing attention experiments.

### 4.3 Agent Self-Attention Critic

Use structured tensors:

```python
forward(
    obs: torch.Tensor,          # [batch, num_agents, obs_dim]
    actions: torch.Tensor,      # [batch, num_agents, action_dim]
    agent_mask: torch.Tensor,   # [batch, num_agents], 1=valid/visible, 0=inactive
) -> torch.Tensor              # [batch, num_agents]
```

This avoids flatten-then-unflatten churn and makes mask semantics explicit.

## 5. MASAC Integration

### 5.1 Model Construction

Select classes from MASAC-specific config:

```text
actor_cls =
    AttentionActorNetwork if config.MASAC_ATTENTION_ACTOR else ActorNetwork

critic_cls =
    CriticNetwork if MASAC_CRITIC_MODE == "mlp"
    LocalAttentionCriticNetwork if MASAC_CRITIC_MODE == "local_attention"
    AgentSelfAttentionCriticNetwork if MASAC_CRITIC_MODE == "agent_self_attention"
```

Track separate metadata:

```text
self.use_attention_actor: bool
self.critic_mode: str
```

Do not derive all attention behavior from `self.use_attention`.

Constructor arguments must also branch by mode:

```python
if critic_mode == "mlp":
    critic = CriticNetwork(
        num_agents * obs_dim,
        num_agents * action_dim,
        num_agents,
    )
elif critic_mode == "local_attention":
    critic = LocalAttentionCriticNetwork(num_agents, obs_dim, action_dim)
elif critic_mode == "agent_self_attention":
    critic = AgentSelfAttentionCriticNetwork(num_agents, obs_dim, action_dim)
```

### 5.2 Critic Routing Helpers

Add routing helpers in `marl_models/masac/masac.py`:

```text
_critic_values(critic, obs_tensor, actions_tensor, agent_mask_tensor)
_target_critic_values(critic, next_obs_tensor, next_actions_tensor, next_agent_mask_tensor)
```

Routing:

```text
if critic_mode in {"mlp", "local_attention"}:
    return critic(obs_tensor.reshape(batch, -1), actions_tensor.reshape(batch, -1))

if critic_mode == "agent_self_attention":
    return critic(obs_tensor, actions_tensor, agent_mask_tensor)
```

For `_optimize_actor`, keep predicted actions structured:

```python
pred_actions_tensor = torch.stack(pred_actions_list, dim=1)
q1_pred = self._critic_values(self.critic_1, obs_tensor, pred_actions_tensor, active_mask_tensor)
q2_pred = self._critic_values(self.critic_2, obs_tensor, pred_actions_tensor, active_mask_tensor)
```

After this change, `_optimize_actor` should no longer accept `obs_flat`; flattening belongs inside `_critic_values` for modes that need it.

### 5.3 Mask Semantics

Use masks consistently:

```text
current critic: active_mask_tensor
actor-step critic: active_mask_tensor
target critic: bootstrap_mask_tensor
```

`bootstrap_mask_tensor` is the next-state visibility/bootstrap mask. For target Q, it is the correct attention mask because the target critic evaluates `next_obs_tensor` and `next_actions_tensor`.

The existing target action masking remains:

```text
next_action_i = next_action_i * bootstrap_mask_i
```

The existing loss/target masking remains:

```text
y_i = reward_i + gamma * target_q_i * bootstrap_mask_i
critic_loss = masked_mean(loss_i, active_mask_i)
actor_loss = masked_mean(alpha * log_prob_i - q_i, active_mask_i)
```

## 6. Checkpoint Metadata

Bump:

```text
CHECKPOINT_VERSION: 1 -> 2
```

because the attention architecture and config semantics change.

Recommended checkpoint fields:

```text
checkpoint_format = "shared_masac_configurable_critic"
checkpoint_version = 2
actor_type = "shared"
critic_type = "shared_vector"
uses_attention = use_attention_actor or critic_mode != "mlp"
uses_attention_actor = self.use_attention_actor
critic_mode = self.critic_mode
```

Keep `critic_type = "shared_vector"` only to mean "shared critic returns a per-agent vector". Do not reinterpret it as "critic must receive flattened vectors"; the new `critic_mode` field carries the actual architecture.

Load should reject checkpoints when:

```text
checkpoint_version != self.CHECKPOINT_VERSION
critic_mode != self.critic_mode
uses_attention_actor != self.use_attention_actor
num_agents mismatch
```

Do not add legacy compatibility for old attention checkpoints unless there is a concrete need.

## 7. Error Handling

Fail fast with `ValueError` for:

```text
unknown MASAC_CRITIC_MODE
checkpoint_format mismatch
obs_dim != config.OBS_DIM_SINGLE for local or agent attention paths
MASAC_AGENT_ATTENTION_DIM % MASAC_AGENT_ATTENTION_HEADS != 0
MASAC_AGENT_ID_DIM <= 0
MASAC_AGENT_ATTENTION_LAYERS < 1
agent self-attention obs rank != 3
agent self-attention actions rank != 3
agent self-attention mask rank != 2
obs/actions/mask batch size mismatch
obs/actions/mask num_agents mismatch
obs trailing dim != self.obs_dim
actions trailing dim != self.action_dim
```

Do not silently infer agent count from global `config.NUM_UAVS`; use runtime `num_agents`.

## 8. File Map

Modify:

```text
config.py
    Add MASAC-specific attention config.

marl_models/masac/agents.py
    Rename/preserve old local attention critic.
    Add AgentSelfAttentionBlock.
    Add AgentSelfAttentionCriticNetwork.
    Delete the old ambiguous AttentionCriticNetwork name.
    Keep actor classes local-only.

marl_models/masac/masac.py
    Select actor and critic independently.
    Add critic routing helpers.
    Update six critic call sites.
    Update checkpoint metadata/version checks.

tests/test_masac.py
    Add config-mode, shape, mask, routing, update, and checkpoint tests.
```

Do not modify the replay buffer or environment observation schema.

## 9. Implementation Plan

### Task 1: Add Mode Selection Tests

**Files:**

- Modify: `tests/test_masac.py`
- Test: `tests/test_masac.py`

- [ ] **Step 1: Add tests for independent actor and critic selection**

Test these modes:

```text
MASAC_ATTENTION_ACTOR=False, MASAC_CRITIC_MODE="mlp"
    actor is ActorNetwork
    critic_1 is CriticNetwork

MASAC_ATTENTION_ACTOR=True, MASAC_CRITIC_MODE="mlp"
    actor is AttentionActorNetwork
    critic_1 is CriticNetwork

MASAC_ATTENTION_ACTOR=False, MASAC_CRITIC_MODE="local_attention"
    actor is ActorNetwork
    critic_1 is LocalAttentionCriticNetwork

MASAC_ATTENTION_ACTOR=False, MASAC_CRITIC_MODE="agent_self_attention"
    actor is ActorNetwork
    critic_1 is AgentSelfAttentionCriticNetwork
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
pytest tests/test_masac.py -q
```

Expected before implementation: failures for missing config names/classes.

### Task 2: Add Config and Preserve Old Critic

**Files:**

- Modify: `config.py`
- Modify: `marl_models/masac/agents.py`
- Test: `tests/test_masac.py`

- [ ] **Step 1: Add MASAC attention config in `config.py`**

Add the fields listed in section 2 near existing MASAC hyperparameters.

- [ ] **Step 2: Rename the old attention critic**

Move the current `AttentionCriticNetwork` implementation to `LocalAttentionCriticNetwork`.

Remove `AttentionCriticNetwork` from imports and exports after tests are updated. Do not leave it as an alias.

- [ ] **Step 3: Keep existing local-attention tests passing**

Update imports and assertions in `tests/test_masac.py` so tests refer to `LocalAttentionCriticNetwork` where they mean the old local entity encoder critic.

### Task 3: Implement Agent Self-Attention Critic

**Files:**

- Modify: `marl_models/masac/agents.py`
- Test: `tests/test_masac.py`

- [ ] **Step 1: Add `AgentSelfAttentionBlock`**

Implement a pre-LN residual block using `nn.MultiheadAttention(batch_first=True)`. Its FFN must be `Linear -> SiLU -> Linear` with hidden width `MASAC_AGENT_ATTENTION_DIM * MASAC_AGENT_ATTENTION_FFN_MULT`.

- [ ] **Step 2: Add `AgentSelfAttentionCriticNetwork`**

Implement structured forward:

```text
obs/actions/mask -> batched AttentionEncoder -> token projection
-> vectorized agent ID embedding concat
-> self-attention blocks with safe_key_padding_mask
-> nan_to_num and zero inactive context rows
-> shared Q head with concat(context_i, h_i, a_i, e_i)
-> [batch, num_agents]
```

- [ ] **Step 3: Add mask behavior tests**

Test that changing an inactive agent's observation/action does not change active agents' Q values when the inactive agent mask is zero.

Add an all-inactive batch-row test and assert the critic output is finite.

### Task 4: Wire MASAC Routing

**Files:**

- Modify: `marl_models/masac/masac.py`
- Test: `tests/test_masac.py`

- [ ] **Step 1: Select actor and critic independently**

Use `MASAC_ATTENTION_ACTOR` and `MASAC_CRITIC_MODE`.

- [ ] **Step 2: Add critic routing helpers**

Route flattened inputs only for `mlp` and `local_attention`; pass structured tensors only for `agent_self_attention`.

- [ ] **Step 3: Replace critic call sites**

Update these six calls:

```text
target_critic_1(next_obs_flat, next_actions_flat)
target_critic_2(next_obs_flat, next_actions_flat)
critic_1(obs_flat, actions_flat)
critic_2(obs_flat, actions_flat)
critic_1(obs_flat, pred_actions_flat)
critic_2(obs_flat, pred_actions_flat)
```

Also remove `obs_flat` from `_optimize_actor` and pass structured `pred_actions_tensor` into `_critic_values`.

### Task 5: Update Checkpoint Tests

**Files:**

- Modify: `marl_models/masac/masac.py`
- Modify: `tests/test_masac.py`

- [ ] **Step 1: Bump checkpoint version to 2**

Set `CHECKPOINT_FORMAT = "shared_masac_configurable_critic"` and reject older format/version checkpoints for this architecture.

- [ ] **Step 2: Save and validate new metadata**

Assert checkpoints include:

```text
uses_attention_actor
critic_mode
```

- [ ] **Step 3: Add mismatch rejection tests**

Reject checkpoint loads when saved `critic_mode` or `uses_attention_actor` differs from the current model.

### Task 6: Verification

**Files:**

- Test: `tests/test_attention_masking.py`
- Test: `tests/test_masac.py`
- Test: `tests`

- [ ] **Step 1: Run focused attention and MASAC tests**

Run:

```powershell
pytest tests/test_attention_masking.py tests/test_masac.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
pytest tests -q
```

Expected: full suite passes.

## 10. Ablation Plan

Run with matched seed sets, episode budgets, and logging windows:

```text
Baseline:
    MASAC_ATTENTION_ACTOR=False
    MASAC_CRITIC_MODE="mlp"

Actor-only local attention:
    MASAC_ATTENTION_ACTOR=True
    MASAC_CRITIC_MODE="mlp"

Critic-only local entity attention:
    MASAC_ATTENTION_ACTOR=False
    MASAC_CRITIC_MODE="local_attention"

Critic-only agent self-attention:
    MASAC_ATTENTION_ACTOR=False
    MASAC_CRITIC_MODE="agent_self_attention"

Full local actor + agent self-attention critic:
    MASAC_ATTENTION_ACTOR=True
    MASAC_CRITIC_MODE="agent_self_attention"
```

Primary metrics:

```text
reward
rate
latency
energy
fairness
collisions
boundaries
```

Training diagnostics:

```text
actor_loss
critic_loss
q_value_mean
alpha_mean
action_std
actor_grad_norm
critic_grad_norm
agent_attention_entropy or max attention weight when easy to log
```

Treat single-seed results as debugging evidence only. Performance claims require seed-averaged comparisons.

## 11. Acceptance Criteria

- `MASAC_CRITIC_MODE="mlp"` preserves existing raw MLP critic behavior.
- `MASAC_CRITIC_MODE="local_attention"` preserves the old local entity encoder critic for ablation.
- `MASAC_CRITIC_MODE="agent_self_attention"` uses masked agent-level self-attention over UAV tokens built from `h_i`, `a_i`, and learned `agent_id_embedding_i`.
- Agent ID embeddings are generated with vectorized `torch.arange(..., device=obs.device)` and broadcasting, not per-agent Python loops.
- The Q head receives `concat(context_i, h_i, a_i, agent_id_embedding_i)` to preserve a direct SAC actor-gradient path through `a_i`.
- Actor attention is controlled independently by `MASAC_ATTENTION_ACTOR`.
- Actor never receives centralized information.
- Target critic uses `bootstrap_mask_tensor` as the next-state attention mask.
- Current and actor-step critics use `active_mask_tensor`.
- Inactive contexts are zeroed after self-attention and before Q head.
- Inactive final Q values are also zeroed after Q head because the Q head receives skip-connected local features.
- All-inactive mask rows are handled without NaNs.
- Twin critics do not share parameters.
- Existing SAC target, alpha update, action squashing, replay buffer, and loss masking semantics remain unchanged.
- Tests cover mode selection, tensor shape contracts, inactive-agent masking, update smoke paths, and checkpoint metadata mismatches.

## 12. Non-Goals

- Do not introduce centralized actor execution.
- Do not share encoders between twin critics.
- Do not change reward shaping.
- Do not change replay buffer schema.
- Do not change environment observation structure.
- Do not add recurrent state.
- Do not retain duplicate or misleading critic class names after the rename.
- Do not keep `AttentionCriticNetwork` as a compatibility alias.
