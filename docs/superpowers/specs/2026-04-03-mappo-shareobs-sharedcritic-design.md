# MAPPO Share-Obs Shared-Critic Alignment Design

Date: 2026-04-03

Status: Proposed

## Goal

Align the local MAPPO implementation with the `on-policy` `share_policy=True` design in the following specific ways:

- one shared actor
- one shared critic
- critic input is flattened centralized observation `share_obs = concat(obs_1, ..., obs_N)`
- critic output is a single scalar value `V(share_obs)`
- remove the current agent-conditioned critic path based on `agent_id`

This design intentionally prioritizes architectural alignment with `on-policy` over preserving the current local `V(joint_obs, agent_id)` semantics.

## Non-Goals

- Do not introduce per-agent actor instances.
- Do not introduce per-agent critic instances.
- Do not add recurrent actor or critic support.
- Do not change the current bounded-action PPO contract based on tanh-squashed Gaussian actions.
- Do not refactor the environment observation construction itself.
- Do not change reward shaping or active/inactive UAV environment logic in this task.

## Current Context

The local repository currently uses:

- one shared actor
- one shared critic
- critic input as structured `joint_obs [batch, num_agents, obs_dim]`
- critic query path conditioned on `agent_index`
- critic outputs per-agent values through `V(joint_obs, active_mask, agent_id)`

The `on-policy` reference with `share_policy=True` instead uses:

- one shared policy object
- one shared actor
- one shared critic
- centralized flattened `share_obs`
- critic outputs one scalar value for each sample of `share_obs`

For MPE-style tasks, `on-policy` constructs `share_obs` by flattening all agents' observations and then repeating that same flattened vector for each agent slot in the rollout buffer.

## Design Summary

Replace the current query-style centralized critic with a standard flattened-share-observation critic:

- actor remains shared and still consumes local per-agent observations
- critic becomes a pure centralized value network over flattened `share_obs`
- rollout storage stores one `share_obs` per environment step
- training minibatches flatten `(time, agent)` samples, and each sample receives the corresponding step's `share_obs`
- remove `agent_id`, `joint_obs_index`, and the current agent-conditioned value head

## Data Contracts

### Actor

Unchanged:

- input: `obs_i` with shape `[batch, obs_dim]`
- output: tanh-squashed Gaussian policy evaluated through corrected log-probabilities

### Critic

New contract:

- input: `share_obs` with shape `[batch, num_agents * obs_dim]`
- output: `values` with shape `[batch]`

For rollout collection at one environment step:

- environment produces `obs_arr` with shape `[num_agents, obs_dim]`
- MAPPO constructs `share_obs = obs_arr.reshape(num_agents * obs_dim)`
- critic evaluates that one flattened centralized observation once
- the resulting scalar value is broadcast to all agent entries for that step in the rollout buffer

This mirrors the official shared-policy pattern where all agents at a step use the same centralized value estimate.

## Network Architecture

### Non-Attention Critic

Use a standard MLP value network:

- input dim: `num_agents * obs_dim`
- `Linear -> LayerNorm -> SiLU`
- `Linear -> LayerNorm -> SiLU`
- `Linear -> LayerNorm -> SiLU`
- `Linear -> 1`

The critic is no longer agent-conditioned.

### Attention Critic

The attention branch must also align with the same scalar-value contract.

Recommended structure:

- reshape flattened `share_obs` back to `[batch, num_agents, obs_dim]` inside the critic
- encode each agent observation with the existing `AttentionEncoder`
- pool encoded agent features with `AgentPoolingValue`
- feed pooled global context into a scalar value head

Important: the attention branch may preserve internal structured processing, but its external interface must still be:

- input: flattened `share_obs`
- output: single scalar value

This keeps the external MAPPO contract aligned while preserving the existing attention encoder investment.

## Rollout Buffer Changes

Current buffer fields to remove from the MAPPO critic path:

- `joint_obs`
- `joint_obs_index`
- `agent_index`
- any logic that deduplicates joint observations per minibatch for critic queries

New storage requirements:

- store one `share_obs` per timestep: `[T, num_agents * obs_dim]`
- keep local per-agent `obs`, `raw_actions`, `old_log_probs`, `rewards`, `active_mask`
- store scalar `value` per step, then replicate it across agents when producing flattened training samples

Mini-batch generation:

- flattened samples remain indexed by `(time, agent)`
- `share_obs_batch` is obtained by indexing timestep-aligned flattened centralized observations
- every agent sample from the same timestep receives the same `share_obs_batch` row

## Training Semantics

### Value Prediction

At collection time:

- compute one scalar critic value from `share_obs`
- assign that same value to all agent entries for the timestep

At update time:

- critic loss uses scalar predictions against per-sample returns
- because returns remain stored per flattened `(time, agent)` sample, the same value prediction is compared against each agent's target from that timestep

This is an intentional consequence of aligning to the standard shared centralized value setup.

### Advantage and Return Computation

No design change to:

- GAE recursion
- global advantage normalization
- PPO actor update logic
- value clipping logic

Only the value-function input/output contract changes.

## Active Mask Handling

Active masks remain relevant for loss masking and GAE bootstrapping.

However, active masks are no longer part of the critic forward signature.

They remain used in:

- rollout buffer GAE recursion
- actor loss masking
- critic loss masking
- metric aggregation

They are no longer used to alter the centralized critic encoding path directly.

## Testing Plan

Add or update tests to cover:

- critic accepts flattened `share_obs` and returns a scalar value
- MAPPO rollout path constructs `share_obs` from `obs.reshape(-1)`
- update path feeds `share_obs_batch` into critic
- buffer emits timestep-aligned `share_obs_batch` for flattened `(time, agent)` samples
- old `agent_index`-based critic API is removed
- attention and non-attention critic branches both obey the same flattened-share-observation contract

Regression tests should explicitly verify that:

- all agents from the same timestep receive the same old value prediction
- critic no longer depends on `agent_id`

## Risks and Tradeoffs

### Pros

- matches `on-policy` shared-policy MAPPO more closely
- simpler critic interface
- easier cross-repo reasoning and hyperparameter comparison
- removes custom query-index machinery from the rollout buffer

### Cons

- loses the current ability to represent per-agent value differences through `agent_id`
- may fit this environment less naturally because rewards include agent-specific penalties
- attention critic becomes externally flatter and less expressive than the current `V(joint_obs, agent_id)` design

This tradeoff is accepted because the explicit goal is architectural alignment with the official shared-policy MAPPO pattern.

## Implementation Boundaries

Files expected to change:

- `marl_models/mappo/agents.py`
- `marl_models/mappo/mappo.py`
- `marl_models/mappo/rollout_buffer.py`
- `tests/test_mappo_attention.py`

Files not to change unless required by a test failure:

- `environment/env.py`
- reward shaping config
- generic actor policy code outside MAPPO

## Acceptance Criteria

The redesign is complete when:

- MAPPO uses one shared actor and one shared critic
- critic input is flattened `share_obs`
- critic output is a single scalar value
- `agent_id` is fully removed from the critic contract
- rollout buffer no longer uses the current joint-query remapping path
- MAPPO tests pass under both attention and non-attention critic branches
- the resulting structure is directly explainable as analogous to `on-policy` `share_policy=True`
