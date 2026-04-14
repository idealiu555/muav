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


def _ue_summary_slice() -> slice:
    start = config.ATTENTION_UAV_EMBED_DIM
    return slice(start, start + config.ATTENTION_EMBED_DIM)


def _neighbor_summary_slice() -> slice:
    start = config.ATTENTION_UAV_EMBED_DIM + config.ATTENTION_EMBED_DIM
    return slice(start, start + config.ATTENTION_NEIGHBOR_DIM)


def test_attention_encoder_outputs_expected_shape_for_zero_count_obs() -> None:
    encoder = AttentionEncoder()
    obs = _make_obs_batch(batch_size=3)

    encoded = encoder(obs)

    assert tuple(encoded.shape) == (3, encoder.output_dim)
    assert torch.isfinite(encoded).all()


def test_attention_encoder_zero_count_obs_zeroes_entity_summary_slices() -> None:
    encoder = AttentionEncoder()
    obs = _make_obs_batch(batch_size=3)

    encoded = encoder(obs)

    assert torch.allclose(encoded[:, _ue_summary_slice()], torch.zeros_like(encoded[:, _ue_summary_slice()]), atol=1e-6)
    assert torch.allclose(
        encoded[:, _neighbor_summary_slice()],
        torch.zeros_like(encoded[:, _neighbor_summary_slice()]),
        atol=1e-6,
    )


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


def test_attention_encoder_mixed_batch_keeps_rows_isolated() -> None:
    encoder = AttentionEncoder()

    zero_obs = _make_obs_batch(batch_size=1)

    populated_obs = _make_obs_batch(batch_size=1)
    neighbor_start = _neighbor_section_start()
    ue_start = _ue_section_start()
    populated_obs[0, _neighbor_count_index()] = 1.0
    populated_obs[0, _ue_count_index()] = 1.0
    populated_obs[0, neighbor_start:neighbor_start + config.NEIGHBOR_STATE_DIM] = torch.arange(
        1, config.NEIGHBOR_STATE_DIM + 1, dtype=torch.float32
    )
    populated_obs[0, ue_start:ue_start + config.UE_STATE_DIM] = torch.arange(
        1, config.UE_STATE_DIM + 1, dtype=torch.float32
    )

    mixed_obs = torch.cat([zero_obs, populated_obs], dim=0)

    mixed_encoded = encoder(mixed_obs)
    zero_encoded = encoder(zero_obs)
    populated_encoded = encoder(populated_obs)

    assert torch.allclose(mixed_encoded[0], zero_encoded[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(mixed_encoded[1], populated_encoded[0], atol=1e-6, rtol=1e-6)
