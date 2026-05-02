# Superseded Design Note

This document is retained only as a historical pointer.

The original `2026-05-01` design described a shared-actor + shared scalar-critic refactor for `MADDPG` and `MASAC` under a team-reward assumption. That design is no longer the active implementation direction.

It was superseded by the vector-critic design in:

- [2026-05-02-shared-actor-vector-critic-design.md](2026-05-02-shared-actor-vector-critic-design.md)

Current code semantics:

- shared actor for all agents;
- shared vector critic outputs one value per agent;
- per-agent rewards are preserved;
- per-agent bootstrap masks are preserved;
- `MASAC` uses masked per-agent entropy terms instead of a summed joint-entropy target.

Do not use the old scalar-critic / team-reward assumptions in this file for further implementation or review.
