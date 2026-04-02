from marl_models.base_model import MARLModel
from marl_models.buffer_and_helpers import ReplayBuffer
from marl_models.mappo.rollout_buffer import MAPPORolloutBuffer
from marl_models.utils import save_models
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot
from utils.plot_logs import generate_plots_if_available

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


def _append_active_actions(action_accumulator: list[np.ndarray], actions: np.ndarray, active_mask: np.ndarray) -> None:
    active_actions = actions[active_mask.astype(bool, copy=False)]
    if active_actions.size > 0:
        action_accumulator.append(active_actions)


def train_on_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    if config.PPO_ROLLOUT_LENGTH != config.STEPS_PER_EPISODE:
        raise ValueError(
            "MAPPO uses full-episode rollouts in this finite-horizon environment. "
            "Set PPO_ROLLOUT_LENGTH == STEPS_PER_EPISODE."
        )
    buffer: MAPPORolloutBuffer = MAPPORolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        buffer_size=config.PPO_ROLLOUT_LENGTH,
        device=model.device,
    )
    num_updates: int = num_episodes
    save_freq: int = max(1, num_updates // 10)
    if num_updates < 1000:
        save_freq = min(100, num_updates)
    print(f"Total updates to be performed: {num_updates}")
    print(f"Each update has {config.PPO_ROLLOUT_LENGTH} steps.")
    print(f"Updates for {config.PPO_EPOCHS} epochs with batch size {config.PPO_BATCH_SIZE}.")
    rollout_log: Log = Log()
    
    # Track training statistics
    training_stats_accumulator: dict = {}
    action_accumulator: list = []

    for update in range(1, num_updates + 1):
        obs: list[np.ndarray] = env.reset()
        rollout_reward: float = 0.0
        rollout_latency: float = 0.0
        rollout_energy: float = 0.0
        rollout_fairness: float = 0.0
        rollout_rate: float = 0.0
        rollout_collisions: int = 0
        rollout_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, update, 0, logger.log_dir, "update", logger.timestamp, True)

        for step in range(1, config.PPO_ROLLOUT_LENGTH + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, update, step, logger.log_dir, "update", logger.timestamp)

            obs_arr: np.ndarray = np.array(obs)
            env_actions, raw_actions, log_probs, values = model.get_action_and_value(obs_arr)

            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, reward_stats, step_collisions, step_boundaries), _step_info = env.step(env_actions)
            _append_active_actions(action_accumulator, env_actions, _step_info["active_mask"])
            # update_trajectories(env)  # tracking code, comment if not needed
            buffer.add(
                obs_arr,
                raw_actions,
                log_probs,
                rewards,
                values,
                _step_info["active_mask"],
            )

            obs = next_obs

            rollout_reward += np.sum(rewards)
            rollout_latency += total_latency
            rollout_energy += total_energy
            rollout_fairness += jfi
            rollout_rate += total_rate
            rollout_collisions += step_collisions
            rollout_boundaries += step_boundaries
            
            for key, value in reward_stats.items():
                if key not in training_stats_accumulator:
                    training_stats_accumulator[key] = []
                training_stats_accumulator[key].append(value)

        stats = model.train_on_rollout(buffer, current_update=update, total_updates=num_updates)
        if stats:
            for key, value in stats.items():
                if key not in training_stats_accumulator:
                    training_stats_accumulator[key] = []
                training_stats_accumulator[key].append(value)

        buffer.clear()

        # Normalize metrics by number of steps for interpretability
        rollout_log.append(
            rollout_reward / config.PPO_ROLLOUT_LENGTH, 
            rollout_latency / config.PPO_ROLLOUT_LENGTH, 
            rollout_energy / config.PPO_ROLLOUT_LENGTH, 
            rollout_fairness / config.PPO_ROLLOUT_LENGTH, 
            rollout_rate / config.PPO_ROLLOUT_LENGTH, 
            rollout_collisions, 
            rollout_boundaries
        )
        if update % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            # Average training statistics
            averaged_stats: dict = {}
            if training_stats_accumulator:
                for key, values in training_stats_accumulator.items():
                    if key == 'noise_scale':
                        averaged_stats[key] = float(values[-1])
                    else:
                        averaged_stats[key] = float(np.mean(values))
                # Add action statistics
                if action_accumulator:
                    all_actions = np.concatenate(action_accumulator, axis=0)
                    averaged_stats["action_mean"] = float(np.mean(all_actions))
                    averaged_stats["action_std"] = float(np.std(all_actions))

            logger.log_metrics(update, rollout_log, config.LOG_FREQ, elapsed_time, "update", averaged_stats)
            rollout_log.keep_recent(config.LOG_FREQ * 2)  # Keep 2x window for safety
            # Reset accumulators
            training_stats_accumulator.clear()
            action_accumulator.clear()
        if update % 100 == 0:
            generate_plots_if_available(logger.json_file_path, f"train_plots/{config.MODEL}/", "train", logger.timestamp)
        if update % save_freq == 0 and update < num_updates:
            save_models(model, update, "update", logger.timestamp)

    save_models(model, -1, "update", logger.timestamp, final=True)


def train_off_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    buffer: ReplayBuffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    save_freq: int = num_episodes // 10
    if num_episodes < 1000:
        save_freq = 100
    episode_log: Log = Log()

    # Track training statistics
    training_stats_accumulator: dict = {}
    update_count: int = 0
    action_accumulator: list = []
    total_step_count: int = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            total_step_count += 1
            if total_step_count <= config.INITIAL_RANDOM_STEPS:
                actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
                active_indices = [i for i, uav in enumerate(env.uavs) if uav.active]
                if active_indices:
                    actions[active_indices] = np.random.uniform(-1, 1, size=(len(active_indices), config.ACTION_DIM))
            else:
                actions = model.select_actions(obs, exploration=True)

            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, reward_stats, step_collisions, step_boundaries), step_info = env.step(actions)
            if total_step_count > config.INITIAL_RANDOM_STEPS:
                _append_active_actions(action_accumulator, actions, step_info["active_mask"])
            # update_trajectories(env)  # tracking code, comment if not needed
            
            for key, value in reward_stats.items():
                if key not in training_stats_accumulator:
                    training_stats_accumulator[key] = []
                training_stats_accumulator[key].append(value)
            
            episode_done: bool = step == config.STEPS_PER_EPISODE
            bootstrap_mask = np.zeros(config.NUM_UAVS, dtype=np.float32) if episode_done else step_info["next_active_mask"]
            buffer.add(obs, actions, rewards, next_obs, active_mask=step_info["active_mask"], bootstrap_mask=bootstrap_mask)

            if total_step_count > config.INITIAL_RANDOM_STEPS and step % config.LEARN_FREQ == 0 and len(buffer) > config.REPLAY_BATCH_SIZE:
                batch = buffer.sample(config.REPLAY_BATCH_SIZE)
                stats = model.update(batch)
                update_count += 1
                # Accumulate training statistics (filter None values to avoid diluting averages)
                for key, value in stats.items():
                    if value is not None:  # Only accumulate non-None values
                        if key not in training_stats_accumulator:
                            training_stats_accumulator[key] = []
                        training_stats_accumulator[key].append(value)

            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness += jfi
            episode_rate += total_rate
            episode_collisions += step_collisions
            episode_boundaries += step_boundaries
            
        # Decay exploration noise at the end of each episode (for MATD3/MADDPG)
        # Only decay after warmup phase to avoid premature exploration reduction
        if hasattr(model, 'noise') and total_step_count > config.INITIAL_RANDOM_STEPS:
            for n in model.noise:
                n.decay()

        # Normalize metrics by number of steps for interpretability
        episode_log.append(
            episode_reward / config.STEPS_PER_EPISODE, 
            episode_latency / config.STEPS_PER_EPISODE, 
            episode_energy / config.STEPS_PER_EPISODE, 
            episode_fairness / config.STEPS_PER_EPISODE, 
            episode_rate / config.STEPS_PER_EPISODE, 
            episode_collisions, 
            episode_boundaries
        )
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            # Average training statistics
            if training_stats_accumulator:
                averaged_stats = {}
                for key, values in training_stats_accumulator.items():
                    if key == 'noise_scale':
                        averaged_stats[key] = float(values[-1])
                    else:
                        averaged_stats[key] = float(np.mean(values))
                # Add action statistics
                if action_accumulator:
                    all_actions = np.concatenate(action_accumulator, axis=0)
                    averaged_stats["action_mean"] = float(np.mean(all_actions))
                    averaged_stats["action_std"] = float(np.std(all_actions))
            else:
                averaged_stats = {}
            
            # Always record buffer_size and update_count (even during warmup)
            averaged_stats["buffer_size"] = len(buffer)
            averaged_stats["update_count"] = update_count
            
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, "episode", averaged_stats)
            episode_log.keep_recent(config.LOG_FREQ * 2)  # Keep 2x window for safety
            # Reset accumulators
            training_stats_accumulator.clear()
            update_count = 0
            action_accumulator.clear()
        if episode % 100 == 0:
            generate_plots_if_available(logger.json_file_path, f"train_plots/{config.MODEL}/", "train", logger.timestamp)
        if episode % save_freq == 0 and episode < num_episodes:
            save_models(model, episode, "episode", logger.timestamp)

    save_models(model, -1, "episode", logger.timestamp, final=True)


def train_random(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            actions: np.ndarray = model.select_actions(obs, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, _reward_stats, step_collisions, step_boundaries), _step_info = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            obs = next_obs

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness += jfi
            episode_rate += total_rate
            episode_collisions += step_collisions
            episode_boundaries += step_boundaries
            
        episode_log.append(episode_reward / config.STEPS_PER_EPISODE, episode_latency / config.STEPS_PER_EPISODE, episode_energy / config.STEPS_PER_EPISODE, episode_fairness / config.STEPS_PER_EPISODE, episode_rate / config.STEPS_PER_EPISODE, episode_collisions, episode_boundaries)
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, "episode")
            episode_log.keep_recent(config.LOG_FREQ * 2)  # Keep 2x window for safety
        if episode % 100 == 0:
            generate_plots_if_available(logger.json_file_path, f"train_plots/{config.MODEL}/", "train", logger.timestamp)
