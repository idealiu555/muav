from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
# 基于强化学习框架，模拟多无人机（UAV）空中基站通信保障环境，管理 UAV、用户设备（UE）和宏基站（MBS）的状态、动作和奖励。


def _min_distance_on_interval(rel_pos: np.ndarray, rel_vel: np.ndarray, duration: float) -> float:
    """Return the minimum distance for synchronous linear relative motion on [0, duration]."""
    if duration <= 0.0:
        return float(np.linalg.norm(rel_pos))

    vel_sq = float(np.dot(rel_vel, rel_vel))
    if vel_sq <= float(config.EPSILON):
        return float(np.linalg.norm(rel_pos))

    t_star = -float(np.dot(rel_pos, rel_vel)) / vel_sq
    t_star = float(np.clip(t_star, 0.0, duration))
    closest_delta = rel_pos + rel_vel * t_star
    return float(np.linalg.norm(closest_delta))


def _synchronous_trajectory_min_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    p_move_time: float,
    q0: np.ndarray,
    q1: np.ndarray,
    q_move_time: float,
) -> float:
    """Return the minimum same-time distance over one slot for move-then-hover trajectories."""
    slot_duration = float(config.TIME_SLOT_DURATION)
    p_vel = np.zeros(3, dtype=np.float64) if p_move_time <= float(config.EPSILON) else (p1 - p0) / p_move_time
    q_vel = np.zeros(3, dtype=np.float64) if q_move_time <= float(config.EPSILON) else (q1 - q0) / q_move_time
    breakpoints = sorted({
        0.0,
        float(np.clip(p_move_time, 0.0, slot_duration)),
        float(np.clip(q_move_time, 0.0, slot_duration)),
        slot_duration,
    })

    min_dist = float("inf")
    for start_t, end_t in zip(breakpoints[:-1], breakpoints[1:]):
        duration = end_t - start_t
        p_pos = p0 + p_vel * min(start_t, p_move_time)
        q_pos = q0 + q_vel * min(start_t, q_move_time)
        interval_p_vel = p_vel if start_t < p_move_time else np.zeros_like(p_vel)
        interval_q_vel = q_vel if start_t < q_move_time else np.zeros_like(q_vel)
        rel_pos = p_pos - q_pos
        rel_vel = interval_p_vel - interval_q_vel
        min_dist = min(min_dist, _min_distance_on_interval(rel_pos, rel_vel, duration))

    return min_dist


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step = 0
        self._prepare_for_next_step()
        return self._get_obs()

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float, dict, int, int], dict[str, np.ndarray]]:
        """Execute one time step of the simulation."""
        self._time_step += 1
        active_mask = np.array([1.0 if uav.active else 0.0 for uav in self._uavs], dtype=np.float32)

        # 0. Apply beam control actions first (affects current slot's communication)
        if config.BEAM_CONTROL_ENABLED:
            self._apply_beam_actions(actions)

        # 1. Calculate co-channel interference AFTER beam actions are applied
        # 确保干扰计算使用的波束方向与信号计算一致
        self._calculate_ue_interference()

        # 2. Process requests using current time slot state
        # 先初始化所有 UAV 的 working_cache，避免竞态条件
        # （协作 UAV 可能在请求处理时修改其他 UAV 的 _working_cache）
        for uav in self._uavs:
            uav.init_working_cache()
        for uav in self._uavs:
            uav.process_requests()

        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()

        # 2. Execute UAV movement (updates _dist_moved)
        self._apply_actions_to_env(actions)

        # 3. Calculate energy (flight energy uses _dist_moved, comm energy uses this slot's time)
        for uav in self._uavs:
            uav.update_energy_consumption()

        rewards, metrics = self._get_rewards_and_metrics()
        self._clear_failed_uav_peer_backlogs()

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 4. Prepare for next time slot
        for ue in self._ues:
            ue.update_position()

        # Count violations before resetting logic checks
        step_collisions = sum(1 for uav in self._uavs if uav.collision_violation)
        step_boundaries = sum(1 for uav in self._uavs if uav.boundary_violation)
        next_active_mask = np.array([1.0 if uav.active else 0.0 for uav in self._uavs], dtype=np.float32)

        for uav in self._uavs:
            uav.reset_for_next_step()

        self._prepare_for_next_step()
        next_obs: list[np.ndarray] = self._get_obs()
        step_info = {
            "active_mask": active_mask,
            "next_active_mask": next_active_mask,
        }
        return next_obs, rewards, metrics + (step_collisions, step_boundaries), step_info

    def _prepare_for_next_step(self) -> None:
        """Prepare environment state for the next time step.
        
        This includes:
        1. Generate UE requests
        2. Associate UEs to UAVs
        3. Set UAV requested files and neighbors
        4. Select collaborators
        5. Count requesting UAVs for bandwidth allocation
        6. Set communication rates
        7. Set frequency counts for caching policy
        """
        # 1. Generate requests for all UEs
        for ue in self._ues:
            ue.generate_request()
        
        # 2. Associate UEs to UAVs based on coverage
        self._associate_ues_to_uavs()
        
        # 3. Set requested files and neighbors for each UAV
        for uav in self._uavs:
            uav.set_current_requested_files()
            uav.set_neighbors(self._uavs)
        
        # 4. Select collaborator for each UAV
        for uav in self._uavs:
            uav.select_collaborator()
        
        # 5. Count how many UAVs selected each UAV as collaborator (for FDM bandwidth allocation)
        for uav in self._uavs:
            if uav.current_collaborator:
                uav.current_collaborator._num_requesting_uavs += 1
        
        # 6. Set communication rates (must be after _num_requesting_uavs is counted)
        for uav in self._uavs:
            uav._set_rates()
        
        # 7. Set frequency counts for GDSF caching policy
        for uav in self._uavs:
            uav.set_freq_counts()

    def _clear_failed_uav_peer_backlogs(self) -> None:
        """Remove cross-UAV backlog entries pointing to permanently failed UAVs."""
        failed_ids = {uav.id for uav in self._uavs if uav.failed}
        if not failed_ids:
            return

        for uav in self._uavs:
            uav._backlog_tx_uav_uav_as_requester = {
                k: v for k, v in uav._backlog_tx_uav_uav_as_requester.items() if k not in failed_ids
            }
            uav._backlog_rx_uav_uav_as_requester = {
                k: v for k, v in uav._backlog_rx_uav_uav_as_requester.items() if k not in failed_ids
            }
            uav._backlog_tx_uav_uav_as_collaborator = {
                k: v for k, v in uav._backlog_tx_uav_uav_as_collaborator.items() if k not in failed_ids
            }
            uav._backlog_rx_uav_uav_as_collaborator = {
                k: v for k, v in uav._backlog_rx_uav_uav_as_collaborator.items() if k not in failed_ids
            }

    def _get_obs(self) -> list[np.ndarray]:
        """Construct local observation vector for each UAV agent.

        新观测结构（支持注意力机制）：
        - Own state: normalized position (3) + cache bitmap (NUM_FILES) + is_active (1)
        - Neighbors (MAX_UAV_NEIGHBORS): features (26) + count (1)
        - Associated UEs (MAX_ASSOCIATED_UES): features (5) + count (1)

        使用固定上限的 UE 列表和 count 字段生成 mask。
        """
        all_obs: list[np.ndarray] = []

        # Normalization constants for 3D positions
        pos_norm = np.array([config.AREA_WIDTH, config.AREA_HEIGHT, config.UE_MAX_ALT])

        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos / pos_norm
            own_cache: np.ndarray = uav.cache.astype(np.float32) if uav.active else np.zeros(config.NUM_FILES, dtype=np.float32)
            own_is_active = np.array([1.0 if uav.active else 0.0], dtype=np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache, own_is_active])

            # Part 2: Neighbors state
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_STATE_DIM))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[:config.MAX_UAV_NEIGHBORS]

            # Pre-calculate current requested files
            my_requested_files = set()
            for ue in uav.current_covered_ues:
                req_type, _, req_id = ue.current_request
                if req_type == 1:
                    my_requested_files.add(req_id)

            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos - uav.pos) / config.UAV_SENSING_RANGE
                # 原始 cache bitmap（让注意力机制学习）
                neighbor_cache: np.ndarray = (neighbor.cache.astype(np.float32)
                                              if neighbor.active else np.zeros(config.NUM_FILES, dtype=np.float32))
                # 预处理特征（编码领域知识）
                immediate_help = 0.0
                complementarity = 0.0
                if neighbor.active:
                    for file_id in my_requested_files:
                        if neighbor.cache[file_id]:
                            immediate_help = 1.0
                            break
                    intersection = np.sum(np.logical_and(uav.cache, neighbor.cache))
                    union = np.sum(np.logical_or(uav.cache, neighbor.cache))
                    similarity = intersection / (union + config.EPSILON)
                    complementarity = 1.0 - similarity
                neighbor_is_active = 1.0 if neighbor.active else 0.0
                # 混合特征: pos(3) + cache(NUM_FILES) + immediate_help(1) + complementarity(1) + is_active(1)
                neighbor_states[i, :] = np.concatenate([
                    relative_pos,
                    neighbor_cache,
                    np.array([immediate_help, complementarity, neighbor_is_active], dtype=np.float32)
                ])

            # 邻居数量（用于生成 mask）
            neighbor_count = np.array([len(neighbors)], dtype=np.float32)

            # Part 3: State of associated UEs (bounded by MAX_ASSOCIATED_UES)
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_STATE_DIM))
            # 按距离排序所有关联的 UE
            all_ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos - u.pos)))
            actual_ue_count = min(len(all_ues), config.MAX_ASSOCIATED_UES)

            for i in range(actual_ue_count):
                ue = all_ues[i]
                delta_pos: np.ndarray = (ue.pos - uav.pos) / config.UAV_COVERAGE_RADIUS
                _, _, req_id = ue.current_request
                norm_file_id: float = req_id / config.NUM_FILES
                cache_hit: float = 1.0 if uav.cache[req_id] else 0.0
                ue_states[i, :] = np.array([delta_pos[0], delta_pos[1], delta_pos[2], norm_file_id, cache_hit], dtype=np.float32)

            # UE 数量（用于生成 mask，仅注意力模式需要）
            ue_count = np.array([actual_ue_count], dtype=np.float32)

            # Part 4: Combine all parts
            # 统一观测结构: [own_state, neighbor_states, neighbor_count, ue_states, ue_count]
            # - attention 模式使用 count 生成 mask
            # - 无 attention 模式使用 count 做均值池化，避免 padding 污染
            obs: np.ndarray = np.concatenate([
                own_state,
                neighbor_states.flatten(),
                neighbor_count,
                ue_states.flatten(),
                ue_count
            ])
            all_obs.append(obs)

        return all_obs

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Apply bounded movement, then evaluate continuous-time safety on active UAV trajectories."""
        current_positions: np.ndarray = np.array([uav.pos for uav in self._uavs], dtype=np.float64)
        next_positions: np.ndarray = current_positions.copy()
        move_times: np.ndarray = np.zeros(config.NUM_UAVS, dtype=np.float64)
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION
        x_min, y_min = 0.0, 0.0
        x_max, y_max = float(config.AREA_WIDTH), float(config.AREA_HEIGHT)
        active_indices: list[int] = [i for i, uav in enumerate(self._uavs) if uav.active]

        if active_indices:
            movement_actions: np.ndarray = np.array(actions[:, :3], dtype=np.float64)
            delta_vec_raw: np.ndarray = movement_actions[active_indices]
            raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)
            clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
            distances: np.ndarray = clipped_magnitude * max_dist
            denom: np.ndarray = raw_magnitude + float(config.EPSILON)
            directions: np.ndarray = delta_vec_raw / denom
            delta_pos: np.ndarray = directions * distances
            proposed_positions: np.ndarray = current_positions[active_indices] + delta_pos

            for local_idx, uav_idx in enumerate(active_indices):
                proposed = proposed_positions[local_idx]
                in_xy_bounds = (x_min <= proposed[0] <= x_max and
                                y_min <= proposed[1] <= y_max)
                in_z_bounds = config.UAV_MIN_ALT <= proposed[2] <= config.UAV_MAX_ALT
                if not (in_xy_bounds and in_z_bounds):
                    self._uavs[uav_idx].boundary_violation = True

            next_positions[active_indices] = np.clip(
                proposed_positions,
                [x_min, y_min, config.UAV_MIN_ALT],
                [x_max, y_max, config.UAV_MAX_ALT]
            )
            actual_distances = np.linalg.norm(next_positions[active_indices] - current_positions[active_indices], axis=1)
            move_times[active_indices] = actual_distances / (config.UAV_SPEED + float(config.EPSILON))
            move_times[active_indices] = np.clip(move_times[active_indices], 0.0, config.TIME_SLOT_DURATION)

        pair_distances: list[tuple[int, int, float, bool]] = []
        collided_indices: set[int] = set()
        unsafe_span = config.UNSAFE_UAV_DISTANCE - config.COLLISION_DISTANCE
        for idx_i, i in enumerate(active_indices):
            for j in active_indices[idx_i + 1:]:
                min_dist = _synchronous_trajectory_min_distance(
                    current_positions[i], next_positions[i], move_times[i],
                    current_positions[j], next_positions[j], move_times[j]
                )
                pair_collided = min_dist <= config.COLLISION_DISTANCE
                pair_distances.append((i, j, min_dist, pair_collided))
                if pair_collided:
                    collided_indices.add(i)
                    collided_indices.add(j)

        for uav_idx in collided_indices:
            self._uavs[uav_idx].mark_failed()

        if unsafe_span > 0.0:
            for i, j, min_dist, pair_collided in pair_distances:
                if pair_collided:
                    continue
                if min_dist < config.UNSAFE_UAV_DISTANCE:
                    ratio = (config.UNSAFE_UAV_DISTANCE - min_dist) / unsafe_span
                    penalty = config.UNSAFE_PROXIMITY_PENALTY * float(np.clip(ratio, 0.0, 1.0))
                    if i not in collided_indices:
                        self._uavs[i].add_proximity_penalty(penalty)
                    if j not in collided_indices:
                        self._uavs[j].add_proximity_penalty(penalty)

        for i, uav in enumerate(self._uavs):
            uav.update_position(next_positions[i])

    def _apply_beam_actions(self, actions: np.ndarray) -> None:
        """Apply beam control actions from the agent.
        
        Actions format: [dx, dy, dz, beam_theta, beam_phi] where beam_* are in [-1, 1]
        
        Two modes:
        - offset: beam angles are offsets from centroid direction
        - absolute: beam angles are absolute values
        """
        for i, uav in enumerate(self._uavs):
            if actions.shape[1] < 5:
                continue  # No beam control in action
            
            beam_action_theta = float(actions[i, 3])
            beam_action_phi = float(actions[i, 4])
            
            if config.BEAM_CONTROL_MODE == "offset":
                # Offset mode: [-1, 1] -> [-BEAM_OFFSET_RANGE, +BEAM_OFFSET_RANGE]
                delta_theta = beam_action_theta * config.BEAM_OFFSET_RANGE
                delta_phi = beam_action_phi * config.BEAM_OFFSET_RANGE
                uav.set_beam_offset(delta_theta, delta_phi)
            else:
                # Absolute mode: [-1, 1] -> [0, 180] for theta (full sphere), [-180, 180] for phi
                theta = (beam_action_theta + 1.0) / 2.0 * 180.0  # [0, 180]
                phi = beam_action_phi * 180.0                     # [-180, 180]
                uav.set_beam_absolute(theta, phi)

    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV using 3D spherical coverage."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                if not uav.active:
                    continue
                # 使用 3D 距离判断球形覆盖范围
                distance_3d: float = float(np.linalg.norm(uav.pos - ue.pos))
                if distance_3d <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance_3d))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _calculate_ue_interference(self) -> None:
        """Calculate co-channel interference for each UE from non-serving UAVs.
        
        对于每个被服务的UE，计算来自所有其他UAV的同频干扰功率总和。
        干扰功率取决于：
        1. 干扰UAV到UE的距离和信道增益
        2. 干扰UAV的波束方向（3D beamforming）
        3. 干扰UAV是否有关联UE（无关联则不发射）
        """
        from environment import comm_model as comms
        
        # 预计算每个UAV的波束方向
        uav_info: list[tuple[np.ndarray, tuple[float, float]]] = []
        for uav in self._uavs:
            beam_dir = uav.get_final_beam_direction()
            uav_info.append((uav.pos, beam_dir))
        
        # 为每个被服务的UE计算干扰
        for serving_uav_idx, uav in enumerate(self._uavs):
            for ue in uav.current_covered_ues:
                total_interference: float = 0.0
                
                # 累加来自所有其他UAV的干扰
                for interferer_idx, (interferer_pos, interferer_beam) in enumerate(uav_info):
                    if interferer_idx == serving_uav_idx:
                        continue  # 跳过服务UAV本身
                    
                    # failed UAV 或无关联UE的 UAV 不发射，不产生干扰
                    if (not self._uavs[interferer_idx].active or
                            len(self._uavs[interferer_idx].current_covered_ues) == 0):
                        continue
                    
                    # 计算该干扰UAV对此UE的全频带干扰功率（保守估计）
                    interference = comms.calculate_interference_power(
                        interferer_pos, ue.pos, interferer_beam
                    )
                    total_interference += interference
                
                ue.interference_power = total_interference

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float, dict]]:
        """Returns reward, metrics, and fixed-scale reward diagnostics."""
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        total_rate: float = sum(uav.total_downlink_rate for uav in self._uavs)
        
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        scaled_latency = total_latency / (config.LATENCY_REWARD_SCALE + config.EPSILON)
        scaled_energy = total_energy / (config.ENERGY_REWARD_SCALE + config.EPSILON)
        scaled_rate = total_rate / (config.RATE_REWARD_SCALE + config.EPSILON)

        r_latency: float = config.ALPHA_1 * np.log1p(scaled_latency)
        r_energy: float = config.ALPHA_2 * np.log1p(scaled_energy)
        r_fairness: float = config.ALPHA_3 * np.clip((jfi - 0.6) * 5.0, -2.0, 2.0)
        r_rate: float = config.ALPHA_RATE * np.log1p(scaled_rate)

        reward: float = r_fairness + r_rate - r_latency - r_energy
        rewards: list[float] = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_FAILURE_PENALTY
            if uav.active:
                rewards[uav.id] -= uav.proximity_penalty
            if uav.active and uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]

        reward_stats = {
            "total_latency": total_latency,
            "total_energy": total_energy,
            "total_rate": total_rate,
            "scaled_latency": scaled_latency,
            "scaled_energy": scaled_energy,
            "scaled_rate": scaled_rate,
            "r_latency": r_latency,
            "r_energy": r_energy,
            "r_fairness": r_fairness,
            "r_rate": r_rate,
            "shared_reward": reward,
        }

        return rewards, (total_latency, total_energy, jfi, total_rate, reward_stats)
