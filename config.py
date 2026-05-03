import numpy as np

# Training Parameters
MODEL: str = "maddpg"  # options: 'maddpg', 'matd3', 'mappo', 'masac', 'random'
SEED: int = 1234  # random seed for reproducibility

# Initialize random state for config parameters to ensure reproducibility
_config_rng = np.random.RandomState(SEED)
STEPS_PER_EPISODE: int = 1000  # total T
LOG_FREQ: int = 10  # episodes
PLOT_FREQ: int = 50  # episodes/updates
IMG_FREQ: int = 500  # steps (increased to reduce I/O overhead)
TEST_LOG_FREQ: int = 10  # episodes (for testing)
TEST_IMG_FREQ: int = 1000  # steps (for testing)

# Simulation Parameters
MBS_POS: np.ndarray = np.array([500.0, 500.0, 300.0])  # (X_mbs, Y_mbs, Z_mbs) in meters
NUM_UAVS: int = 10  # U
NUM_UES: int = 100  # M
AREA_WIDTH: int = 1000  # X_max in meters
AREA_HEIGHT: int = 1000  # Y_max in meters
TIME_SLOT_DURATION: float = 1.0  # tau in seconds
UE_MAX_DIST: int = 20  # d_max^UE in meters (per time slot)
UE_MAX_WAIT_TIME: int = 10  # in time slots

# UE Height Parameters (3D mobility)
UE_MIN_ALT: float = 0.0     # 地面UE高度
UE_MAX_ALT: float = 600.0   # 空中UE最大高度
UE_GROUND_RATIO: float = 0.5  # 地面UE占比（50%地面，50%空中）
UE_AERIAL_MIN_ALT: float = 50.0   # 空中UE最小高度
UE_AERIAL_MAX_ALT: float = 600.0  # 空中UE最大高度

# UAV Parameters
UAV_MIN_ALT: float = 100.0  # H_min in meters (minimum flight altitude)
UAV_MAX_ALT: float = 500.0  # H_max in meters (maximum flight altitude)
UAV_SPEED: int = 30  # v^UAV in m/s (3D speed limit)
UAV_STORAGE_CAPACITY: np.ndarray = _config_rng.choice(np.arange(5 * 10**6, 20 * 10**6, 10**6), size=NUM_UAVS)  # S_u in bytes
UAV_SENSING_RANGE: float = 500.0  # R^sense in meters
UAV_COVERAGE_RADIUS: float = 250.0  # R in meters (3D spherical coverage)
MIN_UAV_SEPARATION: float = 2*UAV_COVERAGE_RADIUS*0.2  # 允许 80% 覆盖重叠
# assert UAV_COVERAGE_RADIUS * 2 <= MIN_UAV_SEPARATION  # 已注释：允许覆盖重叠
assert UAV_SENSING_RANGE >= MIN_UAV_SEPARATION

# Fairness Calculation Parameters
FAIRNESS_WINDOW_SIZE: int = 100  # 公平性计算的滑动窗口大小（最近N步）

# Collision Avoidance and Penalties
UNSAFE_UAV_DISTANCE: float = 50.0
COLLISION_DISTANCE: float = 5.0
UNSAFE_PROXIMITY_PENALTY: float = 4.0
COLLISION_FAILURE_PENALTY: float = 10.0
BOUNDARY_PENALTY: float = 2.0
NON_SERVED_LATENCY_PENALTY: float = 60.0  # penalty in latency for non-served requests
# IMPORTANT : Reconfigurable, should try for various values including : NUM_UAVS - 1 and NUM_UES
MAX_UAV_NEIGHBORS: int = min(4, NUM_UAVS - 1)
# 注意力机制：设置合理的 UE 观测上限（通过注意力聚合处理可变长度）
MAX_ASSOCIATED_UES: int = 50  # 观测的最大 UE 数量（覆盖>99.9%情况）
assert MAX_UAV_NEIGHBORS >= 1 and MAX_UAV_NEIGHBORS <= NUM_UAVS - 1
assert MAX_ASSOCIATED_UES >= 1 and MAX_ASSOCIATED_UES <= NUM_UES

# Attention Mechanism Parameters (注意力机制参数)
USE_ATTENTION: bool = False  # 是否使用注意力机制处理可变长度 UE 列表
ATTENTION_EMBED_DIM: int = 128  # UE 注意力的 embedding 维度 (heads=2, head_dim=64)
ATTENTION_UAV_EMBED_DIM: int = 64  # UAV 状态的 embedding 维度
ATTENTION_NEIGHBOR_DIM: int = 64  # Neighbor 注意力输出维度 (heads=2, head_dim=32)
ATTENTION_NUM_HEADS: int = 2  # 多头注意力的头数
ATTENTION_NUM_LAYERS: int = 4  # 每个实体分支的残差 cross-attention 层数

POWER_MOVE: float = 60.0  # P_move in Watts
POWER_HOVER: float = 40.0  # P_hover in Watts

# Content Parameters (for caching)
NUM_CONTENTS: int = 20  # K - number of content files that can be cached
NUM_FILES: int = NUM_CONTENTS  # Total cacheable files
FILE_SIZES: np.ndarray = _config_rng.randint(10**5, 5 * 10**5, size=NUM_FILES)  # in bytes
REQUEST_MSG_SIZE: int = 100  # 请求消息大小 (bytes)，用于上行链路
BITS_PER_BYTE: int = 8  # 字节转比特的换算系数
ZIPF_BETA: float = 0.6  # beta^Zipf for content popularity

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 10  # T_cache
GDSF_SMOOTHING_FACTOR: float = 0.5  # beta^gdsf

# Communication Parameters
G_CONSTS_PRODUCT: float = 2.2846 * 1.42 * 1e-4  # G_0 * g_0
TRANSMIT_POWER: float = 0.8  # P_uav in Watts (UAV 发射功率)
MBS_TRANSMIT_POWER: float = 20.0  # P_mbs in Watts (MBS 发射功率，宏基站功率远大于 UAV)
RECEIVE_POWER: float = 0.1  # P_rx in Watts (UAV 接收功率)
UE_TRANSMIT_POWER: float = 0.1  # P_ue in Watts (UE 发射功率，用于上行链路)
AWGN: float = 1e-13  # sigma^2
BANDWIDTH_INTER: int = 30 * 10**6  # B^inter in Hz
BANDWIDTH_EDGE: int = 30 * 10**6  # B^edge in Hz
BANDWIDTH_BACKHAUL: int = 40 * 10**6  # B^backhaul in Hz

# Air-to-Ground Channel Model Parameters (ITU-R / 3GPP based)
# Environment types: 'suburban', 'urban', 'dense_urban', 'highrise_urban'
ENVIRONMENT_TYPE: str = 'urban'
# LoS probability parameters: P_LoS = 1 / (1 + a * exp(-b * (theta - a)))
# Parameters for different environments (a, b) from ITU-R P.1410 / 3GPP TR 36.777
LOS_PARAMS: dict = {
    'suburban': (4.88, 0.43),       # 郊区：建筑稀疏，LoS概率高
    'urban': (9.61, 0.16),          # 城市：中等建筑密度
    'dense_urban': (12.08, 0.11),   # 密集城区：建筑密集
    'highrise_urban': (27.23, 0.08) # 高层城区：高楼林立，LoS概率低
}
# Additional path loss for NLoS (dB) - accounts for shadowing and diffraction
NLOS_ADDITIONAL_LOSS_DB: float = 10.0  # 非视距额外损耗

# 3D Beamforming Parameters (3GPP TR 38.901 antenna model)
ENABLE_BEAMFORMING: bool = True          # 启用3D波束赋形
G_MAX_DBI: float = 18.0                  # 最大天线增益 (dBi)
THETA_3DB: float = 30.0                  # 俯仰角3dB波束宽度 (度)
PHI_3DB: float = 30.0                    # 方位角3dB波束宽度 (度)
SLA_DB: float = 20.0                     # 旁瓣衰减上限 (dB)

# Beam Control Parameters (智能体控制波束指向)
BEAM_CONTROL_ENABLED: bool = True        # 是否启用智能体控制波束
BEAM_CONTROL_MODE: str = "absolute"       # "offset": 相对质心偏移, "absolute": 绝对角度
BEAM_OFFSET_RANGE: float = 30.0          # offset模式下的最大偏移范围 (度)

# Model Parameters

# Reward weights for multi-objective optimization
# All weights set to 1.0, relying on SCALE to balance magnitude
ALPHA_1: float = 1.0  # weightage for latency (penalty)
ALPHA_2: float = 1.0  # weightage for energy (penalty)
ALPHA_3: float = 1.0  # weightage for fairness/JFI (reward)
ALPHA_RATE: float = 1.0  # weightage for system throughput (reward)
REWARD_SCALING_FACTOR: float = 0.12  # scaling factor for rewards (归一化后保持原量级)

# Balanced scales to make log1p(scaled) ≈ 1.5-2.0 for each component
LATENCY_REWARD_SCALE: float = NUM_UES * TIME_SLOT_DURATION * 6  # ~600
ENERGY_REWARD_SCALE: float = NUM_UAVS * POWER_HOVER * TIME_SLOT_DURATION * 0.5  # 200
RATE_REWARD_SCALE: float = 1.8e8

# JFI reward parameters (balanced around midpoint)
JFI_BASELINE: float = 0.6  # JFI midpoint: symmetric reward/penalty
JFI_SCALE: float = 4.8    # Moderate scaling for JFI deviation
JFI_CLIP_MIN: float = -2.0  # Narrower clip range
JFI_CLIP_MAX: float = 2.0

# UE state: pos(3) + file_id(1) + cache_hit(1) = 5
UE_STATE_DIM: int = 5
# Own state: pos(3) + cache(NUM_FILES) + is_active(1)
OWN_STATE_DIM: int = 3 + NUM_FILES + 1
# Neighbor state: pos(3) + cache(NUM_FILES) + immediate_help(1) + complementarity(1) + is_active(1)
# 混合方案：保留原始cache bitmap + 预处理特征 + 活跃标记
NEIGHBOR_STATE_DIM: int = 3 + NUM_FILES + 3  # 26 dims
# 观测维度计算（统一包含 count 字段）
# - 基础部分: uav_state + neighbors + ues
# - 额外字段: neighbor_count(1) + ue_count(1)，用于 attention mask 或无 attention 的均值池化
# OBS_DIM: 24+4*26+1+50*5+1 = 380
# 编码器输出维度：UAV(64) + UE_attn(128) + Neighbor_attn(64) = 256 ≈ OBS_DIM(380)
_OBS_BASE_DIM: int = (OWN_STATE_DIM +
                      MAX_UAV_NEIGHBORS * NEIGHBOR_STATE_DIM +
                      MAX_ASSOCIATED_UES * UE_STATE_DIM)
OBS_DIM_SINGLE: int = _OBS_BASE_DIM + 2


# Local environment action contract:
# - all policies must emit normalized bounded actions in [-1, 1]
# - movement actions are normalized control magnitudes, not raw physical velocities
# - beam actions are normalized control signals that the environment maps to angles
# This is intentionally different from the official onpolicy Box-action path, which
# evaluates/executes the same unsquashed action tensor without a local tanh-squash contract.
ACTION_DIM: int = 5 if BEAM_CONTROL_ENABLED else 3  # [dx, dy, dz] 或 [dx, dy, dz, beam_theta, beam_phi]
MLP_HIDDEN_DIM: int = 768  # shared hidden width for actor/critic MLP blocks

ACTOR_LR: float = 1e-4
MADDPG_ACTOR_LR: float = 5e-5
MAPPO_ACTOR_LR: float = 5e-5
MASAC_ACTOR_LR: float = ACTOR_LR
CRITIC_LR: float = 2e-4
DISCOUNT_FACTOR: float = 0.99  # gamma
UPDATE_FACTOR: float = 0.001  # tau
MAX_GRAD_NORM: float = 8.0  # maximum norm for gradient clipping to prevent exploding gradients
LOG_STD_MAX: float = 2  # maximum log standard deviation for stochastic policies
LOG_STD_MIN: float = -20  # minimum log standard deviation for stochastic policies
EPSILON: float = 1e-9  # small value to prevent division by zero

# Off-policy algorithm hyperparameters
REPLAY_BUFFER_SIZE: int = 6 * 10**5  # B，大概包含前500个episode的数据
REPLAY_BATCH_SIZE: int = 1024  # minibatch size (increased from 64 for better GPU utilization)
INITIAL_RANDOM_STEPS: int = 40000  # steps of random actions for exploration
LEARN_FREQ: int = 10  # steps to learn after

# Gaussian Noise Parameters (for MADDPG and MATD3)
INITIAL_NOISE_SCALE: float = 0.2
MIN_NOISE_SCALE: float = 0.05
NOISE_DECAY_RATE: float = 0.997
BEAM_NOISE_RATIO: float = 0.5  # 波束动作噪声相对于位移动作噪声的比例

# MATD3 Specific Hyperparameters
POLICY_UPDATE_FREQ: int = 2  # delayed policy update frequency (TD3 typically uses 2)
TARGET_POLICY_NOISE: float = 0.2  # standard deviation of target policy smoothing noise.
NOISE_CLIP: float = 0.5  # range to clip target policy smoothing noise

# MAPPO Specific Hyperparameters
PPO_ROLLOUT_LENGTH: int = 1000  # number of steps to collect per rollout (Set to STEPS_PER_EPISODE for episodic tasks)
PPO_EPOCHS: int = 6  # number of epochs to run on the collected rollout data (aligned with official MAPPO best practices)
PPO_BATCH_SIZE: int = 512  # size of mini-batches to use during the update step (increased from 64 for better GPU utilization)
PPO_CLIP_EPS: float = 0.2  # clipping parameter (epsilon) for the PPO surrogate objective
PPO_VALUE_CLIP_EPS: float = 0.2  # clipping parameter for value function (can be same or different from policy clip)
PPO_ENTROPY_COEF_START: float = 0.02  # initial entropy coefficient (decays over training to prevent entropy explosion)
PPO_ENTROPY_COEF_END: float = 0.02  # final entropy coefficient
PPO_ENTROPY_MC_SAMPLES: int = 8  # Monte Carlo samples for squashed-policy entropy estimation
PPO_GAE_LAMBDA: float = 0.95  # GAE lambda for lower-variance advantage estimation
PPO_MAX_LOG_RATIO: float = 10.0  # clip log-ratio before exp to avoid numerical spikes
PPO_VALUE_LOSS_COEF: float = 0.5  # coefficient for value function loss (PPO paper default)

# MASAC Specific Hyperparameters
ALPHA_MIN: float = 1e-3  # lower bound for entropy temperature to avoid collapsing exploration entirely
TARGET_ENTROPY_SCALE: float = 0.5  # scales the default SAC target entropy of -action_dim
ALPHA_LR: float = 3e-4  # learning rate for the entropy temperature alpha
