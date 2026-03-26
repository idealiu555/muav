from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.matd3.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, masked_mean
import config
import torch
import torch.nn.functional as F
import numpy as np
import os


class MATD3(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        # Create networks for each agent
        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_1: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.critics_2: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_1: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_2: list[CriticNetwork] = [CriticNetwork(self.total_obs_dim, self.total_action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()

        # Create optimizers
        self.actor_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors]
        self.critic_1_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_1]
        self.critic_2_optimizers: list[torch.optim.AdamW] = [torch.optim.AdamW(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_2]

        # Exploration Noise
        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

        # Delayed Updates Counter
        self.update_counter: int = 0

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        """Selects actions for all agents based on their observations (decentralized execution)."""
        # Batch all observations for better GPU utilization
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_agents):
                if obs_array[i, config.OWN_STATE_DIM - 1] < 0.5:
                    continue
                action: torch.Tensor = self.actors[i](obs_tensor[i:i+1]).squeeze(0)
                if exploration:
                    action_np: np.ndarray = action.cpu().numpy() + self.noise[i].sample()
                else:
                    action_np = action.cpu().numpy()
                actions[i] = np.clip(action_np, -1.0, 1.0)

        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MATD3 expects OffPolicyExperienceBatch (dict)"
        self.update_counter += 1
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
        critic_updates: int = 0
        actor_updates: int = 0

        for agent_idx in range(self.num_agents):
            valid_mask: torch.Tensor = active_mask_tensor[:, agent_idx:agent_idx + 1]
            if valid_mask.sum().item() == 0.0:
                continue
            # Update Critic
            with torch.no_grad():
                # Get next actions from target actors and add clipped noise
                next_actions: list[torch.Tensor] = []
                for i in range(self.num_agents):
                    next_action_i: torch.Tensor = self.target_actors[i](next_obs_tensor[:, i, :])
                    noise: torch.Tensor = torch.randn_like(next_action_i) * config.TARGET_POLICY_NOISE
                    clipped_noise: torch.Tensor = torch.clamp(noise, -config.NOISE_CLIP, config.NOISE_CLIP)
                    masked_next_action = torch.clamp(next_action_i + clipped_noise, -1.0, 1.0) * bootstrap_mask_tensor[:, i:i + 1]
                    next_actions.append(masked_next_action)

                next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)

                # Compute target Q-value using the minimum of the two target critics
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_flat, next_actions_tensor)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_flat, next_actions_tensor)
                target_q_min: torch.Tensor = torch.min(target_q1, target_q2)

                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_bootstrap: torch.Tensor = bootstrap_mask_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_min * agent_bootstrap

            # Update both critic networks
            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_flat, actions_flat)
            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_flat, actions_flat)
            critic_1_loss: torch.Tensor = masked_mean(F.smooth_l1_loss(current_q1, y, reduction="none"), valid_mask)
            critic_2_loss: torch.Tensor = masked_mean(F.smooth_l1_loss(current_q2, y, reduction="none"), valid_mask)

            # Combined critic update for better GPU utilization
            self.critic_1_optimizers[agent_idx].zero_grad(set_to_none=True)
            self.critic_2_optimizers[agent_idx].zero_grad(set_to_none=True)
            combined_critic_loss: torch.Tensor = critic_1_loss + critic_2_loss
            combined_critic_loss.backward()
            c1_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            c2_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step()
            self.critic_2_optimizers[agent_idx].step()

            total_critic_loss += combined_critic_loss.item()
            total_critic_grad_norm += float(max(c1_grad_norm, c2_grad_norm))
            q_values.extend(current_q1[valid_mask.squeeze(1).bool()].detach().cpu().numpy().flatten().tolist())
            critic_updates += 1

        # Delayed Policy and Target Network Updates
        if self.update_counter % config.POLICY_UPDATE_FREQ == 0:
            for agent_idx in range(self.num_agents):
                valid_mask: torch.Tensor = active_mask_tensor[:, agent_idx:agent_idx + 1]
                if valid_mask.sum().item() == 0.0:
                    continue
                # Update Actor using fresh actions from all current actors; only current agent keeps gradients
                updated_actions: list[torch.Tensor] = []
                for i in range(self.num_agents):
                    agent_active = active_mask_tensor[:, i:i + 1]
                    if i == agent_idx:
                        updated_action = self.actors[i](obs_tensor[:, i, :]) * agent_active
                    else:
                        with torch.no_grad():
                            updated_action = self.actors[i](obs_tensor[:, i, :]) * agent_active
                    updated_actions.append(updated_action if i == agent_idx else updated_action.detach())

                pred_actions_flat: torch.Tensor = torch.cat(updated_actions, dim=1)

                for param in self.critics_1[agent_idx].parameters():
                    param.requires_grad_(False)
                actor_q: torch.Tensor = self.critics_1[agent_idx](obs_flat, pred_actions_flat)
                actor_loss: torch.Tensor = -masked_mean(actor_q, valid_mask)
                self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                actor_loss.backward()
                a_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
                self.actor_optimizers[agent_idx].step()
                for param in self.critics_1[agent_idx].parameters():
                    param.requires_grad_(True)

                total_actor_loss += actor_loss.item()
                total_actor_grad_norm += float(a_grad_norm)
                actor_updates += 1

                # Soft update all target networks
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)

        # Use None for metrics not computed in this update (avoids diluting averages)
        critic_normalizer = max(1, critic_updates)
        actor_normalizer = max(1, actor_updates)
        stats = {
            "critic_loss": total_critic_loss / critic_normalizer,
            "critic_grad_norm": total_critic_grad_norm / critic_normalizer,
            "q_value_mean": float(np.mean(q_values)) if q_values else 0.0,
            "noise_scale": self.noise[0].scale,
            "actor_loss": total_actor_loss / actor_normalizer if self.update_counter % config.POLICY_UPDATE_FREQ == 0 else None,
            "actor_grad_norm": total_actor_grad_norm / actor_normalizer if self.update_counter % config.POLICY_UPDATE_FREQ == 0 else None,
        }
        
        return stats

    def _init_target_networks(self) -> None:
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        for i in range(self.num_agents):
            torch.save(
                {
                    "actor": self.actors[i].state_dict(),
                    "critic_1": self.critics_1[i].state_dict(),
                    "critic_2": self.critics_2[i].state_dict(),
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic_1": self.target_critics_1[i].state_dict(),
                    "target_critic_2": self.target_critics_2[i].state_dict(),
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_1_optimizer": self.critic_1_optimizers[i].state_dict(),
                    "critic_2_optimizer": self.critic_2_optimizers[i].state_dict(),
                },
                os.path.join(directory, f"agent_{i}.pth"),
            )
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        with open(update_counter_path, "w") as f:
            f.write(str(self.update_counter))

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            checkpoint: dict = torch.load(agent_path, map_location=self.device)
            self.actors[i].load_state_dict(checkpoint["actor"])
            self.critics_1[i].load_state_dict(checkpoint["critic_1"])
            self.critics_2[i].load_state_dict(checkpoint["critic_2"])
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics_1[i].load_state_dict(checkpoint["target_critic_1"])
            self.target_critics_2[i].load_state_dict(checkpoint["target_critic_2"])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_1_optimizers[i].load_state_dict(checkpoint["critic_1_optimizer"])
            self.critic_2_optimizers[i].load_state_dict(checkpoint["critic_2_optimizer"])
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        if os.path.exists(update_counter_path):
            with open(update_counter_path, "r") as f:
                self.update_counter = int(f.read())
        else:
            self.update_counter = 0
            print(f"⚠️ Update counter file not found: {update_counter_path}. Setting update_counter to 0.")
