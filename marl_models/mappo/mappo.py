from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.mappo.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import masked_mean
import config
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class MAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.state_dim: int = state_dim
        self.active_flag_index: int = config.OWN_STATE_DIM - 1

        # Create networks
        self.actors: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critics: CriticNetwork = CriticNetwork(state_dim, num_agents).to(device)

        # Create optimizers
        self.actor_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.actors.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.critics.parameters(), lr=config.CRITIC_LR)

    @staticmethod
    def _squash_action_and_log_prob(dist: Normal, pre_tanh_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squashed_action: torch.Tensor = torch.tanh(pre_tanh_action)
        # Stable tanh log-det-Jacobian:
        # log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        log_det_jacobian: torch.Tensor = 2.0 * (np.log(2.0) - pre_tanh_action - F.softplus(-2.0 * pre_tanh_action))
        log_probs: torch.Tensor = dist.log_prob(pre_tanh_action) - log_det_jacobian
        log_probs = log_probs.sum(dim=-1)
        return squashed_action, log_probs

    def _get_active_mask(self, obs: np.ndarray) -> torch.Tensor:
        active_mask_np = (obs[:, self.active_flag_index] >= 0.5).astype(np.float32)
        return torch.from_numpy(active_mask_np).to(self.device, non_blocking=True)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        # Convert observations to tensor once (avoid repeated conversions)
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)
        action_mask: torch.Tensor = self._get_active_mask(obs_array).unsqueeze(1)
        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            if exploration:
                pre_tanh_actions: torch.Tensor = dist.sample()
            else:
                pre_tanh_actions = dist.mean
            actions: torch.Tensor = torch.tanh(pre_tanh_actions) * action_mask

        return actions.cpu().numpy()

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample bounded actions and corrected log-probs for PPO."""
        # Use non_blocking for async data transfer
        obs_tensor: torch.Tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device, non_blocking=True)
        state_tensor: torch.Tensor = torch.from_numpy(state.astype(np.float32)).to(self.device, non_blocking=True)
        active_mask: torch.Tensor = self._get_active_mask(obs)
        action_mask: torch.Tensor = active_mask.unsqueeze(1)

        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            pre_tanh_actions: torch.Tensor = dist.sample()
            actions, log_probs = self._squash_action_and_log_prob(dist, pre_tanh_actions)
            actions = actions * action_mask
            pre_tanh_actions = pre_tanh_actions * action_mask
            log_probs = log_probs * active_mask

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            values: torch.Tensor = self.critics(state_tensor).squeeze(0)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy(), pre_tanh_actions.cpu().numpy()

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        pre_tanh_actions_batch: torch.Tensor = batch["pre_tanh_actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        states_batch: torch.Tensor = batch["states"]
        old_values_batch: torch.Tensor = batch["old_values"]
        active_mask_batch: torch.Tensor = batch["active_mask"]
        agent_indices_batch: torch.Tensor = batch["agent_indices"]

        valid_mask: torch.Tensor = active_mask_batch.float()
        valid_mask_bool: torch.Tensor = valid_mask.bool()
        valid_count = int(valid_mask.sum().item())

        if valid_count > 0:
            valid_advantages: torch.Tensor = advantages_batch[valid_mask_bool]
            if valid_count > 1:
                adv_mean: torch.Tensor = valid_advantages.mean()
                adv_std: torch.Tensor = valid_advantages.std(unbiased=False)
                valid_advantages = (valid_advantages - adv_mean) / (adv_std + 1e-8)
            valid_advantages = torch.clamp(valid_advantages, -10.0, 10.0)
            advantages_batch = torch.zeros_like(advantages_batch)
            advantages_batch[valid_mask_bool] = valid_advantages
        else:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "ratio_mean": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "value_mean": 0.0,
                "valid_samples": 0.0,
            }

        # Critic Loss
        batch_size = states_batch.shape[0]
        values_all: torch.Tensor = self.critics(states_batch)  # Shape: (batch_size, num_agents)
        values: torch.Tensor = values_all[torch.arange(batch_size, device=values_all.device), agent_indices_batch]
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), valid_mask)

        # Actor Loss
        dist: Normal = self.actors(obs_batch)
        _, new_log_probs = self._squash_action_and_log_prob(dist, pre_tanh_actions_batch)
        log_ratio: torch.Tensor = torch.clamp(
            new_log_probs - old_log_probs_batch,
            -config.PPO_MAX_LOG_RATIO,
            config.PPO_MAX_LOG_RATIO,
        )
        ratio: torch.Tensor = torch.exp(log_ratio)

        # PPO surrogate loss
        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -masked_mean(torch.min(surr1, surr2), valid_mask)

        # Use pre-tanh entropy by default for lower-variance PPO updates.
        if config.PPO_USE_SQUASHED_ENTROPY:
            entropy_pre_tanh: torch.Tensor = dist.rsample()
            _, entropy_log_probs = self._squash_action_and_log_prob(dist, entropy_pre_tanh)
            entropy: torch.Tensor = masked_mean(-entropy_log_probs, valid_mask)
        else:
            entropy = masked_mean(dist.entropy().sum(dim=-1), valid_mask)
        actor_loss -= config.PPO_ENTROPY_COEF * entropy

        # Combined update: zero gradients once, compute both losses, then step
        # Using set_to_none=True is faster than setting to zero
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        
        # Combined backward pass for better GPU utilization
        total_loss: torch.Tensor = actor_loss + critic_loss
        total_loss.backward()
        
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors.parameters(), config.MAX_GRAD_NORM)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": masked_mean(ratio, valid_mask).item(),
            "approx_kl": masked_mean((old_log_probs_batch - new_log_probs).abs(), valid_mask).item(),
            "clip_fraction": masked_mean((torch.abs(ratio - 1.0) > config.PPO_CLIP_EPS).float(), valid_mask).item(),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "value_mean": masked_mean(values, valid_mask).item(),
            "valid_samples": float(valid_count),
        }

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": self.actors.state_dict(),
                "critic": self.critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
