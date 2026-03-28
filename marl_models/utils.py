from marl_models.base_model import MARLModel
from marl_models.maddpg.maddpg import MADDPG
from marl_models.matd3.matd3 import MATD3
from marl_models.mappo.mappo import MAPPO
from marl_models.masac.masac import MASAC
from marl_models.random_baseline.random_model import RandomModel
import config
import torch
import os
import json


def init_gpu_optimizations() -> None:
    """Initialize GPU optimizations for better performance."""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for Ampere+ GPUs (significant speedup with minimal precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN autotuner to find the best algorithm for your hardware
        torch.backends.cudnn.benchmark = True
        # Disable debug mode for faster execution
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        print("✅ GPU optimizations enabled (TF32, cuDNN benchmark)")


def get_device() -> str:
    """Check if GPU is available and set device accordingly."""
    if torch.cuda.is_available():
        init_gpu_optimizations()
        print("\nFound GPU, using CUDA.\n")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("\nUsing MPS (Apple Silicon GPU).\n")
        return "mps"
    else:
        print("\nNo GPU available, using CPU.\n")
        return "cpu"


def get_model(model_name: str) -> MARLModel:
    device = get_device()
    if model_name == "maddpg":
        return MADDPG(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "matd3":
        return MATD3(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "mappo":
        return MAPPO(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, state_dim=config.STATE_DIM, device=device)
    elif model_name == "masac":
        return MASAC(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    elif model_name == "random":
        return RandomModel(model_name=model_name, num_agents=config.NUM_UAVS, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported types: maddpg, matd3, mappo, masac, random")


def save_models(
    model: MARLModel,
    progress_step: int,
    name: str,
    timestamp: str,
    final: bool = False,
    total_steps: int = 0,
    completed_updates: int | None = None,
):
    save_dir: str = f"saved_models/{model.model_name}_{timestamp}"
    if final:
        save_dir = f"{save_dir}/final"
    else:
        save_dir = f"{save_dir}/{name.lower()}_{progress_step:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(save_dir)

    training_state: dict[str, int] = {"total_steps": int(total_steps)}
    if completed_updates is not None:
        training_state["completed_updates"] = int(completed_updates)
    training_state_path: str = os.path.join(save_dir, "training_state.json")
    with open(training_state_path, "w", encoding="utf-8") as f:
        json.dump(training_state, f, indent=4)

    if final:
        print(f"📁 Final models saved in: {save_dir}\n")
    else:
        print(f"📁 Models saved for {name.lower()} {progress_step} in: {save_dir}\n")


def load_training_state(directory: str) -> dict[str, int]:
    training_state_path: str = os.path.join(directory, "training_state.json")
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"❌ Training state file not found: {training_state_path}")

    with open(training_state_path, "r", encoding="utf-8") as f:
        training_state = json.load(f)

    if "total_steps" not in training_state:
        raise ValueError(f"❌ Training state is missing 'total_steps': {training_state_path}")

    return {
        "total_steps": int(training_state["total_steps"]),
        "completed_updates": int(training_state.get("completed_updates", 0)),
    }
