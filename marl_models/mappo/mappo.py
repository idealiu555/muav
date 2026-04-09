import copy

from marl_models.base_model import MARLModel
from marl_models.mappo.agents import ActorNetwork, AttentionActorNetwork, AttentionCriticNetwork, CriticNetwork
from marl_models.mappo.value_norm import ValueNorm
from marl_models.buffer_and_helpers import masked_mean
import config
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class MAPPO(MARLModel):
    def __init__(
        self,
        model_name: str,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str,
    ) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.active_flag_index: int = config.OWN_STATE_DIM - 1

        actor_cls = AttentionActorNetwork if config.USE_ATTENTION else ActorNetwork
        critic_cls = AttentionCriticNetwork if config.USE_ATTENTION else CriticNetwork
        self.use_attention: bool = actor_cls is AttentionActorNetwork

        self.actors = actor_cls(obs_dim, action_dim).to(device)
        self.critics = critic_cls(num_agents, obs_dim).to(device)

        # Create optimizers
        self.actor_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.actors.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.critics.parameters(), lr=config.CRITIC_LR)
        self.entropy_coef: float = config.PPO_ENTROPY_COEF_START
        self.entropy_mc_samples: int = config.PPO_ENTROPY_MC_SAMPLES
        self.value_normalizer = ValueNorm(1, device=device)

    def _checkpoint_metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "use_attention": self.use_attention,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    def _validate_checkpoint_metadata(self, metadata: object) -> None:
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError("MAPPO checkpoint metadata must be a dict when present")

        expected_values = {
            "model_name": self.model_name,
            "use_attention": self.use_attention,
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        for key, expected in expected_values.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"Incompatible MAPPO checkpoint metadata for {key}: expected {expected!r}, got {actual!r}"
                )

    def _load_checkpoint_state_atomically(
        self,
        actor_state: dict[str, torch.Tensor],
        critic_state: dict[str, torch.Tensor],
        value_normalizer_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        actor_backup = copy.deepcopy(self.actors.state_dict())
        critic_backup = copy.deepcopy(self.critics.state_dict())
        value_norm_backup = copy.deepcopy(self.value_normalizer.state_dict())
        failing_component = "actor"
        try:
            self.actors.load_state_dict(actor_state)
            failing_component = "critic"
            self.critics.load_state_dict(critic_state)
            if value_normalizer_state is not None:
                failing_component = "value_normalizer"
                self.value_normalizer.load_state_dict(value_normalizer_state)
        except RuntimeError as exc:
            self.actors.load_state_dict(actor_backup)
            self.critics.load_state_dict(critic_backup)
            self.value_normalizer.load_state_dict(value_norm_backup)
            error_text = str(exc)
            compatibility_markers = (
                "size mismatch",
                "Missing key(s) in state_dict",
                "Unexpected key(s) in state_dict",
            )
            if any(marker in error_text for marker in compatibility_markers):
                raise ValueError(
                    f"Incompatible MAPPO checkpoint {failing_component} state_dict: {error_text}"
                ) from exc
            raise
        except Exception:
            self.actors.load_state_dict(actor_backup)
            self.critics.load_state_dict(critic_backup)
            self.value_normalizer.load_state_dict(value_norm_backup)
            raise

    def _get_active_mask(self, obs: np.ndarray) -> torch.Tensor:
        active_mask_np = (obs[:, self.active_flag_index] >= 0.5).astype(np.float32)
        return torch.from_numpy(active_mask_np).to(self.device, non_blocking=True)

    def _build_share_obs(self, obs_arr: np.ndarray) -> torch.Tensor:
        expected_obs_shape: tuple[int, int] = (self.num_agents, self.obs_dim)
        if obs_arr.shape != expected_obs_shape:
            raise ValueError(f"Expected obs shape {expected_obs_shape}, got {obs_arr.shape}")
        share_obs = np.ascontiguousarray(obs_arr.reshape(1, -1), dtype=np.float32)
        return torch.from_numpy(share_obs).to(self.device, non_blocking=True)

    def _squash_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Map Gaussian samples to the local environment's bounded action contract."""
        return torch.tanh(raw_actions)

    def _tanh_log_det_jacobian(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return 2.0 * (np.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))

    def _squashed_log_prob_from_raw_actions(
        self, dist: Normal, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self._squash_actions(raw_actions)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1) - self._tanh_log_det_jacobian(raw_actions).sum(dim=-1)
        return actions, log_probs

    def _estimate_policy_entropy(self, dist: Normal, actor_mask: torch.Tensor) -> torch.Tensor:
        # Estimate entropy of the executed tanh-squashed policy, not the unsquashed Gaussian.
        if self.entropy_mc_samples <= 0:
            raise ValueError("PPO_ENTROPY_MC_SAMPLES must be positive")

        entropy_raw_actions: torch.Tensor = dist.rsample((self.entropy_mc_samples,))
        _, entropy_log_probs = self._squashed_log_prob_from_raw_actions(dist, entropy_raw_actions)
        entropy_estimates: torch.Tensor = -entropy_log_probs.mean(dim=0)
        return masked_mean(entropy_estimates, actor_mask)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)
        action_mask: torch.Tensor = self._get_active_mask(obs_array).unsqueeze(1)
        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            if exploration:
                raw_actions: torch.Tensor = dist.sample()
            else:
                raw_actions = dist.mean
            actions: torch.Tensor = self._squash_actions(raw_actions) * action_mask

        return actions.cpu().numpy()

    def get_action_and_value(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions under the local bounded-action contract.

        The local environment expects normalized actions in [-1, 1], so MAPPO uses a
        tanh-squashed Gaussian policy and evaluates log-probs on the executed actions.
        """
        obs_arr: np.ndarray = np.asarray(obs, dtype=np.float32)
        expected_obs_shape: tuple[int, int] = (self.num_agents, self.obs_dim)
        if obs_arr.shape != expected_obs_shape:
            raise ValueError(f"Expected obs shape {expected_obs_shape}, got {obs_arr.shape}")
        if not np.all(np.isfinite(obs_arr)):
            raise ValueError("obs contains NaN or Inf values")

        obs_tensor: torch.Tensor = torch.from_numpy(obs_arr).to(self.device, non_blocking=True)
        share_obs: torch.Tensor = self._build_share_obs(obs_arr)
        active_mask: torch.Tensor = self._get_active_mask(obs)
        action_mask: torch.Tensor = active_mask.unsqueeze(1)

        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            raw_actions: torch.Tensor = dist.sample()
            env_actions, log_probs = self._squashed_log_prob_from_raw_actions(dist, raw_actions)
            env_actions = env_actions * action_mask
            critic_output: torch.Tensor = self.critics(share_obs)
            values = self._critic_values_for_rollout(critic_output)

        return (
            env_actions.cpu().numpy(),
            raw_actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def _compute_entropy_coef(self, *, current_update: int, total_updates: int) -> float:
        progress = min(current_update / max(total_updates, 1), 1.0)
        return config.PPO_ENTROPY_COEF_START - (
            config.PPO_ENTROPY_COEF_START - config.PPO_ENTROPY_COEF_END
        ) * progress

    def _critic_values_for_rollout(self, critic_output: torch.Tensor) -> torch.Tensor:
        if critic_output.ndim == 1:
            if critic_output.shape[0] == 1:
                return critic_output.repeat(self.num_agents)
            if critic_output.shape[0] == self.num_agents:
                return critic_output
        elif critic_output.ndim == 2 and critic_output.shape == (1, self.num_agents):
            return critic_output.squeeze(0)
        raise ValueError(
            f"Unexpected critic rollout output shape {tuple(critic_output.shape)}; "
            f"expected (1,), ({self.num_agents},), or (1, {self.num_agents})"
        )

    def _critic_values_for_batch(self, critic_output: torch.Tensor) -> torch.Tensor:
        if critic_output.ndim == 1:
            return critic_output
        if critic_output.ndim == 2 and critic_output.shape[1] == 1:
            return critic_output.squeeze(1)
        raise ValueError(
            f"Unexpected critic batch output shape {tuple(critic_output.shape)}; "
            "expected (batch,) or (batch, 1)"
        )

    def _update_minibatch(self, batch: dict[str, torch.Tensor]) -> dict:
        assert isinstance(batch, dict), "MAPPO expects dict batch"
        obs_batch: torch.Tensor = batch["obs"]
        raw_actions_batch: torch.Tensor = batch["raw_actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        share_obs_batch: torch.Tensor = batch["share_obs"]
        old_values_batch: torch.Tensor = batch["old_values"]
        active_mask_batch: torch.Tensor = batch["active_mask"]
        actor_mask: torch.Tensor = active_mask_batch.float()
        valid_count = int(actor_mask.sum().item())

        # Early return if no active agents
        if valid_count == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "ratio_mean": 0.0,
                "clip_fraction": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "value_mean": 0.0,
                "valid_samples": 0.0,
                "log_std_mean": 0.0,
                "log_std_std": 0.0,
            }

        # Actor Loss
        dist: Normal = self.actors(obs_batch)
        _, new_log_probs = self._squashed_log_prob_from_raw_actions(dist, raw_actions_batch)
        log_ratio: torch.Tensor = torch.clamp(
            new_log_probs - old_log_probs_batch,
            -config.PPO_MAX_LOG_RATIO,
            config.PPO_MAX_LOG_RATIO,
        )
        ratio: torch.Tensor = torch.exp(log_ratio)

        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -masked_mean(torch.min(surr1, surr2), actor_mask)

        entropy: torch.Tensor = self._estimate_policy_entropy(dist, actor_mask)
        actor_loss = actor_loss - self.entropy_coef * entropy

        # Critic Loss (masked)
        critic_output: torch.Tensor = self.critics(share_obs_batch)
        values: torch.Tensor = self._critic_values_for_batch(critic_output)
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
        active_indices = actor_mask > 0.5
        if active_indices.any():
            self.value_normalizer.update(returns_batch[active_indices].detach())
        normalized_returns = self.value_normalizer.normalize(returns_batch.detach())
        vf_loss1: torch.Tensor = (values - normalized_returns).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - normalized_returns).pow(2)
        critic_loss: torch.Tensor = config.PPO_VALUE_LOSS_COEF * masked_mean(torch.max(vf_loss1, vf_loss2), actor_mask)

        # Actor update
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # Critic update (separate)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # Metrics
        log_std_data: torch.Tensor = self.actors.policy.log_std.data.squeeze(0)
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": masked_mean(ratio, actor_mask).item(),
            "clip_fraction": masked_mean((torch.abs(ratio - 1.0) > config.PPO_CLIP_EPS).float(), actor_mask).item(),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "value_mean": masked_mean(values, actor_mask).item(),
            "valid_samples": float(valid_count),
            "log_std_mean": log_std_data.mean().item(),
            "log_std_std": log_std_data.std().item(),
        }

    def train_on_rollout(self, rollout_buffer: object, *, current_update: int, total_updates: int) -> dict:
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")

        rollout_buffer.compute_returns_and_advantages(
            config.DISCOUNT_FACTOR,
            config.PPO_GAE_LAMBDA,
            value_normalizer=self.value_normalizer,
        )
        rollout_buffer.normalize_advantages()
        self.entropy_coef = self._compute_entropy_coef(current_update=current_update, total_updates=total_updates)

        aggregated_stats: dict[str, float] = {}
        num_minibatch_updates = 0

        for _ in range(config.PPO_EPOCHS):
            for batch in rollout_buffer.get_batches(config.PPO_BATCH_SIZE):
                stats = self._update_minibatch(batch)
                num_minibatch_updates += 1
                for key, value in stats.items():
                    aggregated_stats[key] = aggregated_stats.get(key, 0.0) + float(value)

        if num_minibatch_updates == 0:
            return {"entropy_coef": self.entropy_coef}

        averaged_stats = {key: value / num_minibatch_updates for key, value in aggregated_stats.items()}
        averaged_stats["entropy_coef"] = self.entropy_coef
        return averaged_stats

    def update(self, batch: dict[str, torch.Tensor]) -> dict:
        return self._update_minibatch(batch)

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "metadata": self._checkpoint_metadata(),
                "actor": self.actors.state_dict(),
                "critic": self.critics.state_dict(),
                "value_normalizer": self.value_normalizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device)
        if "actor" not in checkpoint or "critic" not in checkpoint:
            raise ValueError("MAPPO checkpoint must contain both actor and critic state_dict entries")
        self._validate_checkpoint_metadata(checkpoint.get("metadata"))
        self._load_checkpoint_state_atomically(
            checkpoint["actor"],
            checkpoint["critic"],
            checkpoint.get("value_normalizer"),
        )
