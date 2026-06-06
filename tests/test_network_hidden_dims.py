from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.maddpg.agents import ActorNetwork as MADDPGActor
from marl_models.maddpg.agents import CriticNetwork as MADDPGCritic
from marl_models.mappo.agents import ActorNetwork as MAPPOActor
from marl_models.mappo.agents import CriticNetwork as MAPPOCritic
from marl_models.masac.agents import ActorNetwork as MASACActor
from marl_models.masac.agents import AgentSelfAttentionCriticNetwork as AMASACCritic
from marl_models.masac.agents import AttentionActorNetwork as AMASACActor
from marl_models.masac.agents import CriticNetwork as MASACCritic
from marl_models.masac.agents import LocalAttentionCriticNetwork as MASACLocalAttentionCritic
from marl_models.matd3.agents import ActorNetwork as MATD3Actor
from marl_models.matd3.agents import CriticNetwork as MATD3Critic


def test_non_attention_algorithms_share_base_actor_hidden_dim() -> None:
    actors = [
        MADDPGActor(config.OBS_DIM_SINGLE, config.ACTION_DIM),
        MATD3Actor(config.OBS_DIM_SINGLE, config.ACTION_DIM),
        MAPPOActor(config.OBS_DIM_SINGLE, config.ACTION_DIM).policy,
        MASACActor(config.OBS_DIM_SINGLE, config.ACTION_DIM),
    ]

    for actor in actors:
        assert actor.fc1.out_features == config.BASE_ACTOR_HIDDEN_DIM
        assert actor.fc2.out_features == config.BASE_ACTOR_HIDDEN_DIM


def test_non_attention_algorithms_share_base_critic_hidden_dim() -> None:
    total_obs_dim = config.NUM_UAVS * config.OBS_DIM_SINGLE
    total_action_dim = config.NUM_UAVS * config.ACTION_DIM
    critics = [
        MADDPGCritic(total_obs_dim, total_action_dim, config.NUM_UAVS),
        MATD3Critic(total_obs_dim, total_action_dim),
        MAPPOCritic(config.NUM_UAVS, config.OBS_DIM_SINGLE).value_head,
        MASACCritic(total_obs_dim, total_action_dim, config.NUM_UAVS),
    ]

    for critic in critics:
        assert critic.fc1.out_features == config.BASE_CRITIC_HIDDEN_DIM
        assert critic.fc2.out_features == config.BASE_CRITIC_HIDDEN_DIM


def test_attention_algorithms_use_attention_specific_head_dims() -> None:
    amasac_actor = AMASACActor(config.OBS_DIM_SINGLE, config.ACTION_DIM)
    local_attention_critic = MASACLocalAttentionCritic(
        config.NUM_UAVS,
        config.OBS_DIM_SINGLE,
        config.ACTION_DIM,
    )
    amasac_critic = AMASACCritic(config.NUM_UAVS, config.OBS_DIM_SINGLE, config.ACTION_DIM)

    assert amasac_actor.fc1.out_features == config.ATTENTION_ACTOR_HIDDEN_DIM
    assert local_attention_critic.fc1.out_features == config.ATTENTION_CRITIC_HIDDEN_DIM
    assert amasac_critic.q_head[0].out_features == config.ATTENTION_CRITIC_HIDDEN_DIM
