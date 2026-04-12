from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, masked_mean
import config
import torch
import torch.nn as nn
import numpy as np
import os


class MADDPG(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        self.actors = nn.ModuleList(ActorNetwork(obs_dim, action_dim) for _ in range(num_agents)).to(device)
        self.critics = nn.ModuleList(
            CriticNetwork(self.total_obs_dim, self.total_action_dim) for _ in range(num_agents)
        ).to(device)
        self.target_actors = nn.ModuleList(ActorNetwork(obs_dim, action_dim) for _ in range(num_agents)).to(device)
        self.target_critics = nn.ModuleList(
            CriticNetwork(self.total_obs_dim, self.total_action_dim) for _ in range(num_agents)
        ).to(device)
        self._init_target_networks()

        self.actor_optimizers: list[torch.optim.AdamW] = [
            torch.optim.AdamW(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors
        ]
        self.critic_optimizers: list[torch.optim.AdamW] = [
            torch.optim.AdamW(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics
        ]

        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_agents):
                if obs_array[i, config.OWN_STATE_DIM - 1] < 0.5:
                    continue
                action: torch.Tensor = self.actors[i](obs_tensor[i:i + 1]).squeeze(0)
                action_np = action.cpu().numpy()
                if exploration:
                    action_np = action_np + self.noise[i].sample()
                actions[i] = np.clip(action_np, -1.0, 1.0)

        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MADDPG expects OffPolicyExperienceBatch (dict)"
        obs_batch = batch["obs"]
        actions_batch = batch["actions"]
        rewards_batch = batch["rewards"]
        next_obs_batch = batch["next_obs"]
        active_mask_batch = batch["active_mask"]
        bootstrap_mask_batch = batch["bootstrap_mask"]

        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        active_mask_tensor: torch.Tensor = torch.as_tensor(active_mask_batch, dtype=torch.float32, device=self.device)
        bootstrap_mask_tensor: torch.Tensor = torch.as_tensor(bootstrap_mask_batch, dtype=torch.float32, device=self.device)

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        total_actor_loss: float = 0.0
        total_critic_loss: float = 0.0
        total_actor_grad_norm: float = 0.0
        total_critic_grad_norm: float = 0.0
        q_values: list[float] = []
        updated_agents: int = 0

        with torch.no_grad():
            all_actions_detached = [
                self.actors[i](obs_tensor[:, i, :]) * active_mask_tensor[:, i:i + 1]
                for i in range(self.num_agents)
            ]
            next_actions = [
                self.target_actors[i](next_obs_tensor[:, i, :]) * bootstrap_mask_tensor[:, i:i + 1]
                for i in range(self.num_agents)
            ]
            next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)

        active_agent_indices = [
            agent_idx for agent_idx in range(self.num_agents)
            if active_mask_tensor[:, agent_idx].sum().item() > 0.0
        ]

        for agent_idx in active_agent_indices:
            valid_mask: torch.Tensor = active_mask_tensor[:, agent_idx:agent_idx + 1]

            with torch.no_grad():
                target_q_value: torch.Tensor = self.target_critics[agent_idx](next_obs_flat, next_actions_tensor)
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_bootstrap: torch.Tensor = bootstrap_mask_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_value * agent_bootstrap

            current_q_value: torch.Tensor = self.critics[agent_idx](obs_flat, actions_flat)
            critic_loss: torch.Tensor = masked_mean((current_q_value - y).pow(2), valid_mask)
            self.critic_optimizers[agent_idx].zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critics[agent_idx].parameters(),
                config.MAX_GRAD_NORM,
            )
            self.critic_optimizers[agent_idx].step()

            current_joint_actions: list[torch.Tensor] = []
            for i in range(self.num_agents):
                agent_active = active_mask_tensor[:, i:i + 1]
                if i == agent_idx:
                    action = self.actors[i](obs_tensor[:, i, :]) * agent_active
                else:
                    action = all_actions_detached[i]
                current_joint_actions.append(action)

            pred_actions_tensor: torch.Tensor = torch.stack(current_joint_actions, dim=1)
            pred_actions_flat: torch.Tensor = pred_actions_tensor.reshape(batch_size, -1)

            for param in self.critics[agent_idx].parameters():
                param.requires_grad_(False)
            actor_q: torch.Tensor = self.critics[agent_idx](obs_flat, pred_actions_flat)
            actor_loss: torch.Tensor = -masked_mean(actor_q, valid_mask)
            self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actors[agent_idx].parameters(),
                config.MAX_GRAD_NORM,
            )
            self.actor_optimizers[agent_idx].step()
            for param in self.critics[agent_idx].parameters():
                param.requires_grad_(True)

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_actor_grad_norm += float(actor_grad_norm)
            total_critic_grad_norm += float(critic_grad_norm)
            q_values.extend(current_q_value[valid_mask.bool()].detach().cpu().numpy().flatten().tolist())
            updated_agents += 1

        for agent_idx in range(self.num_agents):
            soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
            soft_update(self.target_critics[agent_idx], self.critics[agent_idx], config.UPDATE_FACTOR)

        normalizer = max(1, updated_agents)
        return {
            "actor_loss": total_actor_loss / normalizer,
            "critic_loss": total_critic_loss / normalizer,
            "actor_grad_norm": total_actor_grad_norm / normalizer,
            "critic_grad_norm": total_critic_grad_norm / normalizer,
            "q_value_mean": float(np.mean(q_values)) if q_values else 0.0,
            "q_value_std": float(np.std(q_values)) if q_values else 0.0,
            "noise_scale": self.noise[0].scale,
        }

    def _init_target_networks(self) -> None:
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        for i in range(self.num_agents):
            torch.save(
                {
                    "actor": self.actors[i].state_dict(),
                    "critic": self.critics[i].state_dict(),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic": self.target_critics[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_optimizer": self.critic_optimizers[i].state_dict(),
                    "noise_scale": self.noise[i].scale,
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory not found: {directory}")

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device)

            self.actors[i].load_state_dict(checkpoint["actor"])
            self.critics[i].load_state_dict(checkpoint["critic"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics[i].load_state_dict(checkpoint["target_critic"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizer"])

            if "noise_scale" in checkpoint:
                self.noise[i].scale = checkpoint["noise_scale"]
