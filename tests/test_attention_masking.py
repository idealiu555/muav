from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.attention import AttentionEncoder


def _make_obs_batch(batch_size: int = 1) -> torch.Tensor:
    obs = torch.zeros(batch_size, config.OBS_DIM_SINGLE, dtype=torch.float32)
    obs[:, config.OWN_STATE_DIM - 1] = 1.0
    return obs


def _neighbor_section_start() -> int:
    return config.OWN_STATE_DIM


def _neighbor_count_index() -> int:
    return config.OWN_STATE_DIM + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_STATE_DIM


def _ue_section_start() -> int:
    return _neighbor_count_index() + 1


def _ue_count_index() -> int:
    return _ue_section_start() + config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM


def test_repo_uses_silu_instead_of_legacy_leaky_activation() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    module_name = "Leaky" + "ReLU"
    func_name = "leaky" + "_relu"

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md"}:
            continue
        if "__pycache__" in path.parts:
            continue

        text = path.read_text(encoding="utf-8")
        if module_name in text or func_name in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == [], f"Found leaky ReLU references in: {offenders}"


def test_attention_encoder_outputs_expected_shape_for_zero_count_obs() -> None:
    encoder = AttentionEncoder()
    obs = _make_obs_batch(batch_size=3)

    encoded = encoder(obs)

    assert tuple(encoded.shape) == (3, encoder.output_dim)
    assert torch.isfinite(encoded).all()


def test_attention_encoder_ignores_padded_neighbor_and_ue_values() -> None:
    encoder = AttentionEncoder()
    obs = _make_obs_batch(batch_size=1)

    neighbor_start = _neighbor_section_start()
    ue_start = _ue_section_start()
    obs[0, _neighbor_count_index()] = 1.0
    obs[0, _ue_count_index()] = 1.0

    # Valid first neighbor and first UE entries.
    obs[0, neighbor_start:neighbor_start + config.NEIGHBOR_STATE_DIM] = torch.arange(
        1, config.NEIGHBOR_STATE_DIM + 1, dtype=torch.float32
    )
    obs[0, ue_start:ue_start + config.UE_STATE_DIM] = torch.arange(
        1, config.UE_STATE_DIM + 1, dtype=torch.float32
    )

    padded_modified = obs.clone()
    padded_modified[
        0,
        neighbor_start + config.NEIGHBOR_STATE_DIM:neighbor_start + 2 * config.NEIGHBOR_STATE_DIM,
    ] = 999.0
    padded_modified[
        0,
        ue_start + config.UE_STATE_DIM:ue_start + 2 * config.UE_STATE_DIM,
    ] = 999.0

    encoded = encoder(obs)
    encoded_modified = encoder(padded_modified)

    assert torch.allclose(encoded, encoded_modified, atol=1e-5, rtol=1e-5)
