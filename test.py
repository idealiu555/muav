"""
python main.py test --num_episodes 50 --model_path model_test/model_pth/amappo --config_path model_test/model_config/amappo/amappo.json
"""

import json
import os

from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log
from utils.plot_snapshots import plot_snapshot
from train import _should_capture_artifacts

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


def _compute_test_averages(episode_log: Log) -> dict[str, float]:
    return {
        "reward": float(np.mean(episode_log.rewards)) if episode_log.rewards else 0.0,
        "latency": float(np.mean(episode_log.latencies)) if episode_log.latencies else 0.0,
        "energy": float(np.mean(episode_log.energies)) if episode_log.energies else 0.0,
        "fairness": float(np.mean(episode_log.fairness_scores)) if episode_log.fairness_scores else 0.0,
        "rate": float(np.mean(episode_log.rates)) if episode_log.rates else 0.0,
        "collisions": float(np.mean(episode_log.collisions)) if episode_log.collisions else 0.0,
        "boundaries": float(np.mean(episode_log.boundaries)) if episode_log.boundaries else 0.0,
    }


def _save_test_summary(logger: Logger, episode_log: Log, num_episodes: int) -> None:
    summary_path: str = os.path.join(logger.log_dir, f"test_summary_{logger.timestamp}.json")
    summary = {
        "num_episodes": int(num_episodes),
        "averages": _compute_test_averages(episode_log),
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def test_model(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        capture_images: bool = _should_capture_artifacts(episode)
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_rate: float = 0.0
        episode_collisions: int = 0
        episode_boundaries: int = 0
        # reset_trajectories(env)  # tracking code, comment if not needed
        if capture_images:
            plot_snapshot(env, episode, 0, logger.log_dir, "episode", logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            if capture_images and step % config.TEST_IMG_FREQ == 0:
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
        logger.log_point(
            episode,
            reward=episode_log.rewards[-1],
            latency=episode_log.latencies[-1],
            energy=episode_log.energies[-1],
            fairness=episode_log.fairness_scores[-1],
            rate=episode_log.rates[-1],
            collisions=episode_log.collisions[-1],
            boundaries=episode_log.boundaries[-1],
            name="episode",
            elapsed_time=time.time() - start_time,
        )
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time, "episode")

    _save_test_summary(logger, episode_log, num_episodes)
