# Shared MADDPG / MASAC Design

## Summary

This document specifies how to refactor the current `MADDPG` and `MASAC` implementations so they match the repository's current MAPPO-style parameter sharing direction:

- all agents share one actor;
- all agents share one critic structure and its target counterpart;
- no explicit `agent identity` input is added;
- training assumes all agents already receive the same reward at every timestep.

The implementation order is:

1. refactor `MADDPG`;
2. refactor `MASAC`;
3. keep the network body definitions as stable as possible and only change the ownership and update semantics.

This is a structural algorithm refactor, not just a checkpoint-format cleanup. The actor remains decentralized at execution time because each agent still feeds its own local observation into the shared actor. The critic remains centralized because it still consumes joint observation and joint action.

## Goals

- Make `MADDPG` use a single shared actor, a single shared critic, one target actor, and one target critic.
- Make `MASAC` use a single shared actor, shared twin critics, shared twin target critics, and one shared `log_alpha`.
- Preserve decentralized execution: each agent still computes its own action from its own `obs_i`.
- Preserve centralized training: critics still consume joint state and joint action.
- Aggregate all active-agent learning signals into one update per optimizer step instead of stepping once per agent.
- Enforce the assumption that per-agent rewards are identical rather than silently averaging inconsistent reward vectors.

## Non-Goals

- No introduction of agent identifiers, one-hot role embeddings, or heterogeneous policy heads.
- No change to observation schema, replay buffer schema, or environment reward generation.
- No redesign of the MLP or attention network architectures in `agents.py` beyond what is strictly required for shared ownership.
- No backward compatibility for old per-agent checkpoints.

## Confirmed Assumptions

The design relies on these confirmed assumptions:

- rewards for all agents are already equal at each timestep;
- the desired semantics are fully shared actor and fully shared critic without `agent identity`;
- `MASAC` should also share one temperature parameter `alpha`;
- implementation should proceed with `MADDPG` first and `MASAC` second.

If any of these assumptions changes later, the update targets in this design stop being the correct ones.

## High-Level Semantics

### Shared Actor

Both algorithms move from:

- `a_i = pi_i(o_i)`

to:

- `a_i = pi(o_i)`

The policy parameters are shared, but each agent still provides its own observation. Therefore agents can still produce different actions when their local observations differ.

### Shared Critic

Both algorithms move from per-agent critics to a team critic:

- `Q(s, a_1, ..., a_n)`

This is only logically valid under the confirmed team-reward assumption. The shared critic is not expected to recover distinct per-agent Q-values.

### Team Bootstrap Semantics

Since the critic represents a team value, team bootstrapping should also be team-level. The design therefore uses a single team bootstrap mask per sample:

- `team_bootstrap_mask = min(bootstrap_mask over agent dimension)`

If any agent in the joint transition cannot bootstrap, the full team target does not bootstrap.

## MADDPG Design

### Target Formulation

`MADDPG` will learn a single team Q:

- `a_i = actor(o_i)`
- `a'_i = target_actor(o'_i)`
- `y = r_team + gamma * target_critic(s', a'_1, ..., a'_n) * team_bootstrap_mask`

Where:

- `r_team = rewards[:, 0:1]`
- all other reward columns must match `rewards[:, 0:1]` exactly within a strict tensor equality check

### Critic Update

The shared critic receives:

- flattened joint observation from the sampled batch;
- flattened joint action from replay.

The critic loss is one masked team regression loss:

- `critic_loss = masked_mean((critic(s, a) - y)^2, team_active_mask)`

`team_active_mask` should be derived from the active mask tensor in a way that matches the team interpretation. The recommended choice is:

- `team_active_mask = max(active_mask over agent dimension)`

This keeps a sample trainable when at least one real agent is active, while still zeroing invalid agent actions inside the joint action construction.

### Actor Update

The shared actor is evaluated independently on each local observation in the batch:

- `pred_action_i = actor(obs_i) * active_mask_i`

These predicted actions are concatenated into a single joint action and scored by the shared critic:

- `actor_loss = -masked_mean(critic(s, pred_joint_action), team_active_mask)`

There is one backward pass and one optimizer step for the actor per update call.

### Exploration

Exploration noise remains per-agent runtime state even though actor parameters are shared. This keeps the current execution contract simple and avoids coupling exploration state to policy parameter sharing.

### Save / Load

`MADDPG` will move from per-agent checkpoint files to one checkpoint file containing:

- shared actor state;
- shared critic state;
- target actor state;
- target critic state;
- actor optimizer state;
- critic optimizer state;
- per-agent noise scales;
- checkpoint metadata needed to reject incompatible loads.

## MASAC Design

### Shared Soft-Q Formulation

`MASAC` will learn a team soft Q with one shared stochastic actor, one shared twin-critic pair, and one shared temperature:

- `a_i ~ pi(o_i)`
- `Q1(s, a_1, ..., a_n)`
- `Q2(s, a_1, ..., a_n)`

### Joint Log-Probability

Because the critic scores a joint action, entropy regularization should also be applied at the joint-policy level. The design therefore defines:

- `log_pi_joint = sum_i log pi(a_i | o_i)`

This same joint log-probability definition must be used consistently in:

- the target soft value;
- the actor loss;
- the alpha loss.

### Target Formulation

The target becomes:

- `target_q = min(target_critic_1, target_critic_2) - alpha * log_pi_joint_next`
- `y = r_team + gamma * target_q * team_bootstrap_mask`

Where:

- `r_team = rewards[:, 0:1]`
- all reward columns must match exactly, otherwise training raises an error.

### Critic Update

Both shared critics regress to the same team target:

- `critic_1_loss = masked_mean(smooth_l1(Q1(s, a), y), team_active_mask)`
- `critic_2_loss = masked_mean(smooth_l1(Q2(s, a), y), team_active_mask)`

There is one optimizer step for each shared critic per update call.

### Actor Update

For the current batch:

- the shared actor samples one action and one log-probability per agent;
- all sampled actions are combined into `pred_joint_action`;
- all per-agent log-probabilities are summed into `log_pi_joint`.

The actor loss becomes:

- `actor_loss = masked_mean(alpha * log_pi_joint - min(Q1, Q2), team_active_mask)`

There is one shared actor optimizer step per update call.

### Alpha Update

`MASAC` replaces the current `ParameterList` of `log_alphas` with one shared `log_alpha`.

The target entropy must also switch from single-agent scale to team scale:

- `target_entropy = -(num_agents * action_dim) * TARGET_ENTROPY_SCALE`

The alpha loss becomes:

- `alpha_loss = -masked_mean(log_alpha * (log_pi_joint + target_entropy).detach(), team_active_mask)`

This keeps the entropy target aligned with the joint action space actually regularized by the shared actor.

### Save / Load

`MASAC` will move to one checkpoint file containing:

- shared actor state;
- shared critic 1 state;
- shared critic 2 state;
- shared target critic 1 state;
- shared target critic 2 state;
- shared `log_alpha`;
- actor optimizer state;
- critic 1 optimizer state;
- critic 2 optimizer state;
- alpha optimizer state;
- checkpoint metadata needed to reject incompatible loads.

## File-Level Change Plan

### `marl_models/maddpg/maddpg.py`

- Replace `ModuleList` ownership with single shared network instances.
- Replace per-agent optimizer lists with single optimizers.
- Rewrite `select_actions` to reuse the same actor for each `obs_i`.
- Rewrite `update` to:
  - validate identical rewards across agents;
  - derive team masks;
  - compute one target critic value;
  - compute one critic loss;
  - compute one actor loss;
  - soft-update one target actor and one target critic.
- Replace per-agent save/load with one checkpoint format.

### `marl_models/masac/masac.py`

- Replace `ModuleList` ownership with single shared actor and shared twin critics.
- Replace per-agent `log_alphas` with one `log_alpha`.
- Replace optimizer lists with single optimizers.
- Rewrite `select_actions` to reuse the same actor for each `obs_i`.
- Rewrite `update` to:
  - validate identical rewards across agents;
  - derive team masks;
  - build joint next actions and joint next log-probability;
  - compute one shared twin-critic target;
  - compute one shared actor loss;
  - compute one shared alpha loss;
  - soft-update shared target twins.
- Simplify helper methods that currently depend on `agent_idx`.
- Replace save/load with one checkpoint format.

### `tests/test_maddpg_standard.py`

- Update construction tests to expect one actor, one critic, one target actor, and one target critic.
- Add a reward-consistency validation test.
- Add a shared-update smoke test.
- Update save/load tests for one-checkpoint format.

### `tests/test_masac.py`

- Update construction tests to expect one actor, one shared twin-critic pair, and one shared `log_alpha`.
- Add a test that `target_entropy` scales with `num_agents * action_dim`.
- Add a reward-consistency validation test.
- Add a shared-update smoke test covering joint log-probability aggregation.
- Update save/load tests for one-checkpoint format.

## Error Handling

The refactor should fail fast in these cases:

- reward columns are not identical across agents;
- old per-agent checkpoints are loaded into the new shared-parameter implementation;
- batch shapes do not match expected joint dimensions;
- `MASAC` alpha configuration is invalid;
- team mask reduction produces a shape that no longer matches critic outputs.

Error messages should be explicit about the shared-team assumptions rather than generic shape mismatches whenever possible.

## Testing Strategy

### MADDPG

Required tests:

- constructor creates one shared actor/critic pair and one shared target pair;
- `select_actions` still returns per-agent action tensors with the expected shape;
- reward inconsistency raises a clear error;
- update runs successfully on a representative batch;
- save/load round-trip preserves model and optimizer state.

### MASAC

Required tests:

- constructor creates one shared actor, one shared twin-critic pair, shared targets, and one shared `log_alpha`;
- team target entropy scales correctly with `num_agents * action_dim`;
- reward inconsistency raises a clear error;
- update runs successfully on a representative batch;
- joint log-probability aggregation path is exercised;
- save/load round-trip preserves model and optimizer state.

### Verification Scope

After implementation, verification should include:

- targeted pytest runs for the updated algorithm tests;
- at least one short smoke training run per algorithm if the repository already has a lightweight way to execute one.

## Compatibility and Migration

- Old `MADDPG` per-agent checkpoints are intentionally incompatible with the new shared format.
- Old `MASAC` checkpoints with per-agent actors, critics, and alphas are intentionally incompatible with the new shared format.
- Load errors should clearly state that the implementation now expects shared-parameter checkpoints.

## Risks and Tradeoffs

### Accepted Tradeoffs

- The shared actor assumes agents are exchangeable enough that observation differences alone are sufficient to produce specialized behavior.
- The shared critic gives up agent-specific Q estimation in exchange for simpler team-level training semantics.
- Exploration remains per-agent noise state in `MADDPG`, which is a pragmatic runtime choice rather than a strict theoretical requirement.

### Main Risks

- If the environment later stops producing identical rewards across agents, the shared-critic target becomes semantically invalid.
- Logging semantics will change because losses are no longer averages over independent per-agent optimizers.
- Existing checkpoints and any tooling that expects one file per agent will need to be updated or rejected.

## Implementation Order

1. Refactor `MADDPG` shared ownership and shared update path.
2. Update `MADDPG` tests and verify them.
3. Refactor `MASAC` shared ownership, shared twin critics, and shared `log_alpha`.
4. Update `MASAC` tests and verify them.
5. Review any checkpoint-loading call sites for assumptions about the old file layout.

## Decision

The approved implementation approach is to directly convert both algorithms to shared-parameter team-value variants:

- `MADDPG`: one actor, one critic, one target actor, one target critic;
- `MASAC`: one actor, two shared critics, two shared target critics, one shared `log_alpha`;
- no `agent identity`;
- strict enforcement of identical rewards across agents;
- implementation order: `MADDPG` first, `MASAC` second.
