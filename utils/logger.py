import config as default_config
import json
import os
import numpy as np


class Log:
    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.latencies: list[float] = []
        self.energies: list[float] = []
        self.fairness_scores: list[float] = []
        self.rates: list[float] = []
        self.collisions: list[int] = []
        self.boundaries: list[int] = []

    def append(self, reward: float, latency: float, energy: float, fairness: float, rate: float, collisions: int, boundaries: int) -> None:
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.energies.append(energy)
        self.fairness_scores.append(fairness)
        self.rates.append(rate)
        self.collisions.append(collisions)
        self.boundaries.append(boundaries)
    
    def keep_recent(self, keep_size: int) -> None:
        """Keep only the most recent entries to prevent memory growth."""
        if len(self.rewards) > keep_size:
            self.rewards = self.rewards[-keep_size:]
            self.latencies = self.latencies[-keep_size:]
            self.energies = self.energies[-keep_size:]
            self.fairness_scores = self.fairness_scores[-keep_size:]
            self.rates = self.rates[-keep_size:]
            self.collisions = self.collisions[-keep_size:]
            self.boundaries = self.boundaries[-keep_size:]


class Logger:
    def __init__(self, log_dir: str, timestamp: str) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.timestamp: str = timestamp
        self.log_dir: str = log_dir
        self.json_file_path: str = os.path.join(self.log_dir, f"log_data_{timestamp}.json")
        self.debug_json_file_path: str = os.path.join(self.log_dir, f"debug_data_{timestamp}.json")
        self.config_file_path: str = os.path.join(self.log_dir, f"config_{timestamp}.json")

    def log_configs(self) -> None:
        config_dict: dict = {key: getattr(default_config, key) for key in dir(default_config) if key.isupper() and not key.startswith("__") and not callable(getattr(default_config, key))}

        # Custom serializer for numpy arrays
        def numpy_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(self.config_file_path, "w") as f:
            json.dump(config_dict, f, indent=4, default=numpy_encoder)
        print(f"📝 Configs saved to {self.config_file_path}")

    def load_configs(self, config_path: str) -> None:
        """Load configs for testing - uses saved config from training."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if isinstance(getattr(default_config, key, None), np.ndarray):
                setattr(default_config, key, np.array(value))
            else:
                setattr(default_config, key, value)
        print(f"✅ Configs loaded from {config_path}")

    def log_point(
        self,
        progress_step: int,
        reward: float,
        latency: float,
        energy: float,
        fairness: float,
        rate: float,
        collisions: int,
        boundaries: int,
        name: str,
        elapsed_time: float | None = None,
    ) -> None:
        data_entry: dict = {
            name.lower(): progress_step,
            "reward": float(reward),
            "latency": float(latency),
            "energy": float(energy),
            "fairness": float(fairness),
            "rate": float(rate),
            "collisions": int(collisions),
            "boundaries": int(boundaries),
        }
        if elapsed_time is not None:
            data_entry["time"] = float(elapsed_time)

        with open(self.json_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_entry) + "\n")

    def log_debug_metrics(
        self,
        progress_step: int,
        metrics: dict,
        name: str,
        elapsed_time: float | None = None,
    ) -> None:
        data_entry: dict = {
            name.lower(): progress_step,
        }
        if elapsed_time is not None:
            data_entry["time"] = float(elapsed_time)
        data_entry.update(metrics)

        with open(self.debug_json_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_entry) + "\n")

    def log_metrics(self, progress_step: int, log: Log, log_freq: int, elapsed_time: float, name: str) -> None:
        rewards_slice: np.ndarray = np.array(log.rewards[-log_freq:])
        latencies_slice: np.ndarray = np.array(log.latencies[-log_freq:])
        energies_slice: np.ndarray = np.array(log.energies[-log_freq:])
        fairness_slice: np.ndarray = np.array(log.fairness_scores[-log_freq:])
        rates_slice: np.ndarray = np.array(log.rates[-log_freq:])

        reward_avg: float = float(np.mean(rewards_slice))
        reward_std: float = float(np.std(rewards_slice))
        latency_avg: float = float(np.mean(latencies_slice))
        latency_std: float = float(np.std(latencies_slice))
        energy_avg: float = float(np.mean(energies_slice))
        energy_std: float = float(np.std(energies_slice))
        fairness_avg: float = float(np.mean(fairness_slice))
        rate_avg: float = float(np.mean(rates_slice))
        rate_std: float = float(np.std(rates_slice))
        collisions_latest: int = int(log.collisions[-1]) if log.collisions else 0
        boundaries_latest: int = int(log.boundaries[-1]) if log.boundaries else 0

        log_msg: str = (
            f"🔄 {name.title()} {progress_step} | "
            f"Team Reward: {reward_avg:.3f}±{reward_std:.3f} | "
            f"Lat: {latency_avg:.1f}±{latency_std:.1f} | "
            f"Eng: {energy_avg:.1f}±{energy_std:.1f} | "
            f"JFI: {fairness_avg:.3f} | "
            f"Rate: {rate_avg:.1f}±{rate_std:.1f} | "
            f"Col: {collisions_latest} | "
            f"Bnd: {boundaries_latest} | "
            f"Time: {elapsed_time:.2f}s\n"
        )
        print(log_msg, end="")
