from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot
import config
import numpy as np
import time


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


def test_model(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()
    evaluation_stats_accumulator: dict = {}

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
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
            if step % config.TEST_IMG_FREQ == 0:
                plot_snapshot(env, episode, step, logger.log_dir, "episode", logger.timestamp)

            actions: np.ndarray = model.select_actions(obs, exploration=False)
            next_obs, rewards, (total_latency, total_energy, jfi, total_rate, reward_stats, step_collisions, step_boundaries), step_info = env.step(actions)

            for key, value in reward_stats.items():
                if key not in evaluation_stats_accumulator:
                    evaluation_stats_accumulator[key] = []
                evaluation_stats_accumulator[key].append(value)

            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness += jfi
            episode_rate += total_rate

            episode_collisions += step_collisions
            episode_boundaries += step_boundaries
            episode_steps += 1

            if bool(step_info["terminated"]):
                fleet_failed = bool(step_info["fleet_failed"])
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
        episode_log.append(
            reward_avg,
            latency_avg,
            energy_avg,
            fairness_avg,
            rate_avg,
            episode_collisions, 
            episode_boundaries
        )
        for key, value in {
            "episode_steps": float(episode_steps),
            "survival_ratio": float(episode_steps / config.STEPS_PER_EPISODE),
            "fleet_failure": float(fleet_failed),
        }.items():
            if key not in evaluation_stats_accumulator:
                evaluation_stats_accumulator[key] = []
            evaluation_stats_accumulator[key].append(value)
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            averaged_stats: dict | None = None
            if evaluation_stats_accumulator:
                averaged_stats = {key: float(np.mean(values)) for key, values in evaluation_stats_accumulator.items()}
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time, "episode", averaged_stats)
            evaluation_stats_accumulator.clear()
