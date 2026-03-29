from marl_models.base_model import MARLModel
from marl_models.buffer_and_helpers import RolloutBuffer, ReplayBuffer
from marl_models.utils import save_models
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot
from utils.plot_logs import generate_plots_if_available
import config
import numpy as np
import time


def _append_active_actions(action_accumulator: list[np.ndarray], actions: np.ndarray, active_mask: np.ndarray) -> None:
    active_actions = actions[active_mask.astype(bool, copy=False)]
    if active_actions.size > 0:
        action_accumulator.append(active_actions)


def _append_training_stats(stats_accumulator: dict, stats: dict | None) -> None:
    if not stats:
        return
    for key, value in stats.items():
        if value is None:
            continue
        if key not in stats_accumulator:
            stats_accumulator[key] = []
        stats_accumulator[key].append(float(value))


def _normalize_episode_metrics(
    total_reward: float,
    total_latency: float,
    total_energy: float,
    total_fairness: float,
    total_rate: float,
    episode_steps: int,
) -> tuple[float, float, float, float, float]:
    divisor = max(1, episode_steps)
    return (
        total_reward / divisor,
        total_latency / divisor,
        total_energy / divisor,
        total_fairness / divisor,
        total_rate / divisor,
    )


def _record_episode_diagnostics(stats_accumulator: dict, episode_steps: int, fleet_failed: bool) -> None:
    diagnostics = {
        "episode_steps": float(episode_steps),
        "survival_ratio": float(episode_steps / config.STEPS_PER_EPISODE),
        "fleet_failure": float(fleet_failed),
    }
    for key, value in diagnostics.items():
        if key not in stats_accumulator:
            stats_accumulator[key] = []
        stats_accumulator[key].append(value)


def train_on_policy(
    env: Env,
    model: MARLModel,
    logger: Logger,
    num_episodes: int,
    completed_updates: int = 0,
    total_step_count: int = 0,
) -> None:
    start_time: float = time.time()
    if config.PPO_ROLLOUT_LENGTH != config.STEPS_PER_EPISODE:
        raise ValueError(
            "MAPPO uses full-episode rollouts in this finite-horizon environment. "
            "Set PPO_ROLLOUT_LENGTH == STEPS_PER_EPISODE."
        )
    buffer: RolloutBuffer = RolloutBuffer(num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, state_dim=config.STATE_DIM, buffer_size=config.PPO_ROLLOUT_LENGTH, device=model.device)
    num_updates: int = num_episodes
    save_freq: int = max(1, num_updates // 10)
    if num_updates < 1000:
        save_freq = min(100, num_updates)
    print(f"Total updates to be performed: {num_updates}")
    print(f"Each update runs for up to {config.PPO_ROLLOUT_LENGTH} steps.")
    print(f"Updates for {config.PPO_EPOCHS} epochs with batch size {config.PPO_BATCH_SIZE}.")
    rollout_log: Log = Log()
    
    # Track training statistics
    training_stats_accumulator: dict = {}
    action_accumulator: list = []

    for local_update in range(1, num_updates + 1):
        update: int = completed_updates + local_update
        obs: list[np.ndarray] = env.reset()
        state: np.ndarray = np.concatenate(obs, axis=0)
        rollout_reward: float = 0.0
        rollout_latency: float = 0.0
        rollout_energy: float = 0.0
        rollout_fairness: float = 0.0
        rollout_rate: float = 0.0
        rollout_collisions: int = 0
        rollout_boundaries: int = 0
        rollout_steps: int = 0
        fleet_failed: bool = False
        plot_snapshot(env, update, 0, logger.log_dir, "update", logger.timestamp, True)

        for step in range(1, config.PPO_ROLLOUT_LENGTH + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, update, step, logger.log_dir, "update", logger.timestamp)

            obs_arr: np.ndarray = np.array(obs)
            actions, log_probs, values, pre_tanh_actions = model.get_action_and_value(obs_arr, state)

            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, reward_stats, step_collisions, step_boundaries), _step_info = env.step(actions)
            _append_active_actions(action_accumulator, actions, _step_info["active_mask"])
            env_terminated: bool = bool(_step_info["terminated"])
            episode_done: bool = env_terminated or step == config.PPO_ROLLOUT_LENGTH
            bootstrap_mask: np.ndarray = (
                np.zeros(config.NUM_UAVS, dtype=np.float32)
                if episode_done else _step_info["next_active_mask"]
            )
            buffer.add(
                state,
                obs_arr,
                pre_tanh_actions,
                log_probs,
                rewards,
                values,
                _step_info["active_mask"],
                bootstrap_mask,
            )
            rollout_steps += 1

            rollout_reward += np.sum(rewards)
            rollout_latency += total_latency
            rollout_energy += total_energy
            rollout_fairness += jfi
            rollout_rate += total_rate
            rollout_collisions += step_collisions
            rollout_boundaries += step_boundaries
            _append_training_stats(training_stats_accumulator, reward_stats)

            if env_terminated:
                fleet_failed = bool(_step_info["fleet_failed"])
                break

            obs = next_obs
            state = np.concatenate(next_obs, axis=0)

        buffer.compute_returns_and_advantages(config.DISCOUNT_FACTOR, config.PPO_GAE_LAMBDA)

        epochs_used: int = 0
        early_stop_triggered: bool = False
        for _ in range(config.PPO_EPOCHS):
            epoch_kls: list[float] = []
            for batch in buffer.get_batches(config.PPO_BATCH_SIZE):
                stats = model.update(batch)
                _append_training_stats(training_stats_accumulator, stats)
                if stats and "approx_kl" in stats:
                    epoch_kls.append(float(stats["approx_kl"]))
            epochs_used += 1
            if epoch_kls and float(np.mean(epoch_kls)) > config.PPO_TARGET_KL:
                early_stop_triggered = True
                break

        buffer.clear()
        total_step_count += rollout_steps
        _append_training_stats(training_stats_accumulator, {
            "ppo_epochs_used": float(epochs_used),
            "ppo_early_stop": 1.0 if early_stop_triggered else 0.0,
        })

        reward_avg, latency_avg, energy_avg, fairness_avg, rate_avg = _normalize_episode_metrics(
            rollout_reward,
            rollout_latency,
            rollout_energy,
            rollout_fairness,
            rollout_rate,
            rollout_steps,
        )
        rollout_log.append(
            reward_avg,
            latency_avg,
            energy_avg,
            fairness_avg,
            rate_avg,
            rollout_collisions, 
            rollout_boundaries
        )
        _record_episode_diagnostics(training_stats_accumulator, rollout_steps, fleet_failed)
        if update % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            # Average training statistics
            averaged_stats: dict | None = None
            if training_stats_accumulator:
                averaged_stats = {}
                for key, values in training_stats_accumulator.items():
                    if key == 'noise_scale':
                        averaged_stats[key] = float(values[-1])
                    else:
                        averaged_stats[key] = float(np.mean(values))
                if hasattr(model, "entropy_coef"):
                    averaged_stats["entropy_coef"] = float(model.entropy_coef)
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
        if local_update % save_freq == 0 and local_update < num_updates:
            save_models(
                model,
                update,
                "update",
                logger.timestamp,
                total_steps=total_step_count,
                completed_updates=update,
            )

    save_models(
        model,
        -1,
        "update",
        logger.timestamp,
        final=True,
        total_steps=total_step_count,
        completed_updates=completed_updates + num_updates,
    )


def train_off_policy(env: Env, model: MARLModel, logger: Logger, num_episodes: int, total_step_count: int) -> None:
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

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        episode_steps: int = 0
        fleet_failed: bool = False
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
            
            _append_training_stats(training_stats_accumulator, reward_stats)
            
            env_terminated: bool = bool(step_info["terminated"])
            episode_done: bool = env_terminated or step == config.STEPS_PER_EPISODE
            bootstrap_mask = (
                np.zeros(config.NUM_UAVS, dtype=np.float32)
                if episode_done else step_info["next_active_mask"]
            )
            buffer.add(
                obs,
                actions,
                rewards,
                next_obs,
                active_mask=step_info["active_mask"],
                next_action_mask=step_info["next_active_mask"],
                bootstrap_mask=bootstrap_mask,
            )

            if (
                total_step_count > config.INITIAL_RANDOM_STEPS
                and total_step_count % config.LEARN_FREQ == 0
                and len(buffer) > config.REPLAY_BATCH_SIZE
            ):
                batch = buffer.sample(config.REPLAY_BATCH_SIZE)
                stats = model.update(batch)
                update_count += 1
                _append_training_stats(training_stats_accumulator, stats)

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness += jfi
            episode_rate += total_rate
            episode_collisions += step_collisions
            episode_boundaries += step_boundaries
            episode_steps += 1

            if env_terminated:
                fleet_failed = bool(step_info["fleet_failed"])
                break

            obs = next_obs
            
        # Decay exploration noise at the end of each episode (for MATD3/MADDPG)
        # Only decay after warmup phase to avoid premature exploration reduction
        if hasattr(model, 'noise') and total_step_count > config.INITIAL_RANDOM_STEPS:
            for n in model.noise:
                n.decay()

        reward_avg, latency_avg, energy_avg, fairness_avg, rate_avg = _normalize_episode_metrics(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_rate,
            episode_steps,
        )
        episode_log.append(
            reward_avg,
            latency_avg,
            energy_avg,
            fairness_avg,
            rate_avg,
            episode_collisions, 
            episode_boundaries
        )
        _record_episode_diagnostics(training_stats_accumulator, episode_steps, fleet_failed)
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
            save_models(model, episode, "episode", logger.timestamp, total_steps=total_step_count)

    save_models(model, -1, "episode", logger.timestamp, final=True, total_steps=total_step_count)


def train_random(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()
    training_stats_accumulator: dict = {}

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        episode_steps: int = 0
        fleet_failed: bool = False
        plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if step % config.IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            actions: np.ndarray = model.select_actions(obs, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, reward_stats, step_collisions, step_boundaries), _step_info = env.step(actions)

            _append_training_stats(training_stats_accumulator, reward_stats)

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness += jfi
            episode_rate += total_rate
            episode_collisions += step_collisions
            episode_boundaries += step_boundaries
            episode_steps += 1

            if bool(_step_info["terminated"]):
                fleet_failed = bool(_step_info["fleet_failed"])
                break

            obs = next_obs
            
        reward_avg, latency_avg, energy_avg, fairness_avg, rate_avg = _normalize_episode_metrics(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_rate,
            episode_steps,
        )
        episode_log.append(reward_avg, latency_avg, energy_avg, fairness_avg, rate_avg, episode_collisions, episode_boundaries)
        _record_episode_diagnostics(training_stats_accumulator, episode_steps, fleet_failed)
        if episode % config.LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            averaged_stats: dict | None = None
            if training_stats_accumulator:
                averaged_stats = {key: float(np.mean(values)) for key, values in training_stats_accumulator.items()}
            logger.log_metrics(episode, episode_log, config.LOG_FREQ, elapsed_time, "episode", averaged_stats)
            episode_log.keep_recent(config.LOG_FREQ * 2)  # Keep 2x window for safety
            training_stats_accumulator.clear()
        if episode % 100 == 0:
            generate_plots_if_available(logger.json_file_path, f"train_plots/{config.MODEL}/", "train", logger.timestamp)
