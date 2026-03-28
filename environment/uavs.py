from __future__ import annotations
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np
# 模拟无人机（UAV）作为空中基站提供通信保障服务。它管理 UAV 的位置、缓存、通信、能量消耗和内容请求处理。


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    used_space: int = np.sum(uav._working_cache * config.FILE_SIZES)
    if used_space + config.FILE_SIZES[file_id] <= config.UAV_STORAGE_CAPACITY[uav.id]:
        uav._working_cache[file_id] = True


class UAV:
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array([
            np.random.uniform(0, config.AREA_WIDTH),
            np.random.uniform(0, config.AREA_HEIGHT),
            np.random.uniform(config.UAV_MIN_ALT, config.UAV_MAX_ALT)
        ])

        self._dist_moved: float = 0.0  # Distance moved in the current time slot
        self._current_covered_ues: list[UE] = []
        self._neighbors: list[UAV] = []
        self._current_collaborator: UAV | None = None
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot
        self.failed: bool = False  # Permanently inactive for the rest of the episode once collided
        self._failed_this_step: bool = False  # Distinguish current-slot crash from historical failure
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds
        self._proximity_penalty: float = 0.0  # Continuous safety penalty accumulated in current slot

        # Cache and request tracking
        self._current_requested_files: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)  # For GDSF caching policy
        self._ema_scores = np.zeros(config.NUM_FILES)

        # Communication rates
        self._uav_uav_rate: float = 0.0
        self._uav_mbs_uplink_rate: float = 0.0   # UAV → MBS (发送请求)
        self._uav_mbs_downlink_rate: float = 0.0 # MBS → UAV (接收数据)
        
        # 被多少个 UAV 选为协作者（用于 UAV-UAV 带宽分配）
        self._num_requesting_uavs: int = 0
        
        # Communication time tracking for energy calculation (分为发送和接收)
        # 注意：多用户并行传输（OFDMA），记录的是最大传输时间，不是累加
        # UE ↔ UAV 链路
        self._tx_time_ue_uav: float = 0.0      # UAV 发送时间 (UAV→UE 下行数据)
        self._rx_time_ue_uav: float = 0.0      # UAV 接收时间 (UE→UAV 上行请求)
        # UAV ↔ UAV 链路（作为请求方时，向协作者发送/接收）- 按协作者ID分组
        self._tx_time_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> tx_time
        self._rx_time_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> rx_time
        # UAV ↔ UAV 链路（作为协作者时，来自不同请求方的时间，FDM并行）
        self._tx_time_uav_uav_as_collaborator: dict[int, float] = {}  # requester_id -> tx_time (发数据给请求方)
        self._rx_time_uav_uav_as_collaborator: dict[int, float] = {}  # requester_id -> rx_time (收请求从请求方)
        # UAV ↔ MBS 链路
        self._tx_time_uav_mbs: float = 0.0     # UAV 发送时间 (UAV→MBS 上行请求)
        self._rx_time_uav_mbs: float = 0.0     # UAV 接收时间 (MBS→UAV 下行数据)
        
        # 跨时隙积压时间（上一时隙未完成的传输需要的额外时间）
        # 这些变量跨时隙保持，只在 episode reset 时清零
        # 积压也分为发送和接收
        self._backlog_tx_ue_uav: float = 0.0
        self._backlog_rx_ue_uav: float = 0.0
        # UAV-UAV 作为请求方的积压（按协作者ID分组，因为每时隙可能选择不同协作者）
        self._backlog_tx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> backlog
        self._backlog_rx_uav_uav_as_requester: dict[int, float] = {}  # collaborator_id -> backlog
        # UAV-UAV 作为协作者的积压（FDM并行，按请求方ID分组）
        self._backlog_tx_uav_uav_as_collaborator: dict[int, float] = {}
        self._backlog_rx_uav_uav_as_collaborator: dict[int, float] = {}
        self._backlog_tx_uav_mbs: float = 0.0
        self._backlog_rx_uav_mbs: float = 0.0
        
        # 3D Beamforming: 波束指向 (俯仰角, 方位角)
        self._beam_direction: tuple[float, float] = (0.0, 0.0)  # 基准方向（指向UE质心）
        self._beam_offset: tuple[float, float] = (0.0, 0.0)     # 智能体控制的偏移量
        
        # 传输速率记录（用于奖励计算）
        self._total_downlink_rate: float = 0.0  # 本时隙总下行传输速率

    @property
    def energy(self) -> float:
        return self._energy_current_slot

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list[UAV]:
        return self._neighbors

    @property
    def total_downlink_rate(self) -> float:
        return self._total_downlink_rate

    @property
    def current_collaborator(self) -> UAV | None:
        return self._current_collaborator

    @property
    def active(self) -> bool:
        return not self.failed

    @property
    def proximity_penalty(self) -> float:
        return self._proximity_penalty

    @property
    def mean_associated_latency(self) -> float:
        if not self._current_covered_ues:
            return 0.0
        return float(np.mean([ue.latency_current_request for ue in self._current_covered_ues]))

    def add_proximity_penalty(self, penalty: float) -> None:
        self._proximity_penalty += penalty

    def clear_proximity_penalty(self) -> None:
        self._proximity_penalty = 0.0

    def mark_failed(self) -> None:
        if self.failed:
            return
        self.failed = True
        self._failed_this_step = True
        self.collision_violation = True
        self.clear_proximity_penalty()

    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new time slot.
        
        注意：正常 UAV 的 _backlog_* 变量不在这里重置，因为它们是跨时隙的。
        failed UAV 会在这里清空积压，因为它们已永久退出服务流程。
        """
        self._current_covered_ues = []
        self._neighbors = []
        self._current_collaborator = None
        self._current_requested_files = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)
        self._energy_current_slot = 0.0
        # 本时隙新增的通信时间（每时隙重置，分为发送和接收）
        self._tx_time_ue_uav = 0.0
        self._rx_time_ue_uav = 0.0
        # UAV-UAV 作为请求方（按协作者ID分组）
        self._tx_time_uav_uav_as_requester = {}
        self._rx_time_uav_uav_as_requester = {}
        # UAV-UAV 作为协作者（FDM并行，按请求方ID分组）
        self._tx_time_uav_uav_as_collaborator = {}
        self._rx_time_uav_uav_as_collaborator = {}
        self._tx_time_uav_mbs = 0.0
        self._rx_time_uav_mbs = 0.0
        if self.failed:
            # failed UAV 永久退出服务，清空所有积压、避免影响后续统计
            self._backlog_tx_ue_uav = 0.0
            self._backlog_rx_ue_uav = 0.0
            self._backlog_tx_uav_uav_as_requester = {}
            self._backlog_rx_uav_uav_as_requester = {}
            self._backlog_tx_uav_uav_as_collaborator = {}
            self._backlog_rx_uav_uav_as_collaborator = {}
            self._backlog_tx_uav_mbs = 0.0
            self._backlog_rx_uav_mbs = 0.0
        # 活跃 UAV 的 _backlog_* 跨时隙保持，不在此重置
        self._num_requesting_uavs = 0  # 重置被选为协作者的计数
        self._beam_offset = (0.0, 0.0)  # 重置波束偏移
        self._total_downlink_rate = 0.0  # 重置传输速率记录
        self.collision_violation = False
        self.boundary_violation = False
        self._proximity_penalty = 0.0
        self._failed_this_step = False

    def update_position(self, next_pos: np.ndarray) -> None:
        """Update the UAV's position to the new 3D location chosen by the MARL agent."""
        # next_pos is now a full 3D position [x, y, z]
        self._dist_moved = float(np.linalg.norm(next_pos - self.pos))
        self.pos = next_pos

    def set_neighbors(self, all_uavs: list[UAV]) -> None:
        """Set neighboring UAVs within sensing range for this UAV."""
        self._neighbors = []
        if self.failed:
            return
        for other_uav in all_uavs:
            if other_uav.id != self.id and other_uav.active:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    self._neighbors.append(other_uav)

    def set_current_requested_files(self) -> None:
        """Update the current requested files and beam direction based on covered UEs."""
        if self.failed:
            return
        for ue in self._current_covered_ues:
            if ue.current_request:
                _, _, req_id = ue.current_request
                self._current_requested_files[req_id] = True
        
        # 更新波束基准方向（指向关联UE的质心）
        # 需要计算基准方向的情况：
        # 1. offset 模式（智能体基于此进行调整）
        # 2. 智能体控制被禁用（完全依赖规则，即指向质心）
        if config.BEAM_CONTROL_MODE == "offset" or not config.BEAM_CONTROL_ENABLED:
            ue_positions = [ue.pos for ue in self._current_covered_ues]
            self._beam_direction = comms.calculate_beam_direction(self.pos, ue_positions)

    def set_beam_offset(self, delta_theta: float, delta_phi: float) -> None:
        """Set beam offset from agent's action (offset mode)."""
        self._beam_offset = (delta_theta, delta_phi)

    def set_beam_absolute(self, theta: float, phi: float) -> None:
        """Set beam direction directly from agent's action (absolute mode)."""
        # 在absolute模式下，直接覆盖基准方向，偏移为0
        self._beam_direction = (theta, phi)
        self._beam_offset = (0.0, 0.0)

    def get_final_beam_direction(self) -> tuple[float, float]:
        """Get final beam direction combining base direction and offset.
        
        球坐标系：
        - theta: [0°, 180°]，0°=天顶，90°=水平，180°=天底
        - phi: [-180°, 180°]，方位角
        """
        base_theta, base_phi = self._beam_direction
        delta_theta, delta_phi = self._beam_offset
        
        # 计算最终角度（theta 范围扩展到 [0, 180]）
        final_theta = np.clip(base_theta + delta_theta, 0.0, 180.0)
        # 方位角周期性处理 [-180, 180]
        final_phi = ((base_phi + delta_phi + 180.0) % 360.0) - 180.0
        
        return (final_theta, final_phi)

    def select_collaborator(self) -> None:
        """Choose a single collaborating UAV from its list of neighbours.
        
        注意：_set_rates() 已移至 env.py 中统一调用，
        因为需要先统计所有 UAV 的 _num_requesting_uavs。
        """
        if self.failed:
            self._current_collaborator = None
            return
        if not self._neighbors:
            return

        best_collaborators: list[UAV] = []
        missing_requested_files: np.ndarray = self._current_requested_files & (~self.cache)

        # Hard gate: only collaborate when there is at least one missing file.
        # If all requested files are already cached locally, collaboration is unnecessary.
        if not np.any(missing_requested_files):
            self._current_collaborator = None
            return

        max_missing_overlap: int = -1

        # Find neighbors with maximum overlap
        for neighbor in self._neighbors:
            overlap: int = int(np.sum(missing_requested_files & neighbor.cache))
            if overlap > max_missing_overlap:
                max_missing_overlap = overlap
                best_collaborators = [neighbor]
            elif overlap == max_missing_overlap:
                best_collaborators.append(neighbor)

        # Hard gate: if no neighbor can provide any missing file, do not collaborate.
        # This avoids the extra UAV->UAV->MBS hop that only increases delay.
        if max_missing_overlap <= 0:
            self._current_collaborator = None
            return

        # If only one best collaborator, select it
        if len(best_collaborators) == 1:
            self._current_collaborator = best_collaborators[0]
            return

        # If tie in overlap, select closest one(s)
        min_distance: float = float("inf")
        closest_collaborators: list[UAV] = []

        for collaborator in best_collaborators:
            distance: float = float(np.linalg.norm(self.pos - collaborator.pos))

            if distance < min_distance:
                min_distance = distance
                closest_collaborators = [collaborator]
            elif distance == min_distance:
                closest_collaborators.append(collaborator)

        # If still tied, select randomly
        if len(closest_collaborators) == 1:
            self._current_collaborator = closest_collaborators[0]
        else:
            self._current_collaborator = closest_collaborators[np.random.randint(0, len(closest_collaborators))]

    def set_freq_counts(self) -> None:
        """Set the request count for current slot based on cache availability."""
        if self.failed:
            return
        for ue in self._current_covered_ues:
            _, _, req_id = ue.current_request
            self._freq_counts[req_id] += 1
            if not self.cache[req_id] and self._current_collaborator:
                self._current_collaborator._freq_counts[req_id] += 1

    def init_working_cache(self) -> None:
        """Initialize working cache from current cache state.
        
        必须在所有 UAV 的 process_requests() 之前统一调用，
        避免协作缓存更新时的竞态条件。
        """
        self._working_cache = self.cache.copy()

    def process_requests(self) -> None:
        """Process content requests from UEs covered by this UAV."""
        if self.failed:
            return
        final_beam = self.get_final_beam_direction()
        for ue in self._current_covered_ues:
            channel_gain = comms.calculate_channel_gain(ue.pos, self.pos, final_beam)
            num_ues = len(self._current_covered_ues)
            # 下行速率：UAV → UE，考虑同频干扰（SINR）
            downlink_rate = comms.calculate_ue_uav_rate(channel_gain, num_ues, ue.interference_power)
            # 上行速率：UE → UAV，使用 UE 发射功率（上行干扰暂不考虑）
            uplink_rate = comms.calculate_ue_uav_uplink_rate(channel_gain, num_ues)
            # 记录下行速率用于奖励计算
            self._total_downlink_rate += downlink_rate
            self._process_content_request(ue, downlink_rate, uplink_rate)

    def _set_rates(self) -> None:
        """Set communication rates for UAV-MBS and UAV-UAV links.
        
        UAV-MBS 链路区分上下行：
        - 上行 (UAV→MBS): 使用 UAV 发射功率
        - 下行 (MBS→UAV): 使用 MBS 发射功率（远大于 UAV）
        
        UAV-UAV 链路采用频分复用(FDM)：
        - 如果一个 UAV 被多个邻居选为协作者，其 UAV-UAV 带宽需要平分
        - 带宽分割发生在协作者端，请求方端不分割
        """
        if self.failed:
            self._uav_uav_rate = 0.0
            self._uav_mbs_uplink_rate = 0.0
            self._uav_mbs_downlink_rate = 0.0
            return
        mbs_channel_gain = comms.calculate_channel_gain(self.pos, config.MBS_POS)
        self._uav_mbs_uplink_rate = comms.calculate_uav_mbs_uplink_rate(mbs_channel_gain)
        self._uav_mbs_downlink_rate = comms.calculate_uav_mbs_downlink_rate(mbs_channel_gain)
        
        if self._current_collaborator:
            channel_gain = comms.calculate_channel_gain(self.pos, self._current_collaborator.pos)
            
            # UAV-UAV 链路速率由协作者的带宽分割决定
            # 协作者被多个 UAV 选中时，需要将 BANDWIDTH_INTER 平分给各请求方
            collaborator_load = max(1, self._current_collaborator._num_requesting_uavs)
            self._uav_uav_rate = comms.calculate_uav_uav_rate(channel_gain, collaborator_load)

    def _process_content_request(self, ue: UE, ue_uav_downlink_rate: float, ue_uav_uplink_rate: float) -> None:
        """Process a content request from a UE.
        
        通信流程（双向）:
        1. 上行: UE → UAV (请求消息，UAV 接收)
        2. 上行: UAV → UAV/MBS (请求转发，UAV 发送)
        3. 下行: MBS/UAV → UAV (数据返回，UAV 接收)
        4. 下行: UAV → UE (数据传输，UAV 发送)
        
        能耗模型区分发送和接收功率:
        - 发送：使用 TRANSMIT_POWER
        - 接收：使用 RECEIVE_POWER
        """
        _, _, req_id = ue.current_request

        # 注意单位转换：FILE_SIZES 和 REQUEST_MSG_SIZE 单位是 bytes，
        # 而通信速率（香农公式结果）单位是 bits/s，因此需要乘以 8
        file_size_bits = config.FILE_SIZES[req_id] * config.BITS_PER_BYTE
        request_size_bits = config.REQUEST_MSG_SIZE * config.BITS_PER_BYTE
        
        # === UE ↔ UAV 链路 ===
        # 上行请求：UE → UAV，UAV 接收
        ue_uav_request_latency = request_size_bits / ue_uav_uplink_rate
        # 下行数据：UAV → UE，UAV 发送
        ue_uav_data_latency = file_size_bits / ue_uav_downlink_rate
        ue_uav_transmission_time = ue_uav_request_latency + ue_uav_data_latency
        
        # 分别记录发送和接收时间（OFDMA 并行取最大值）
        self._rx_time_ue_uav = max(self._rx_time_ue_uav, ue_uav_request_latency)  # 接收请求
        self._tx_time_ue_uav = max(self._tx_time_ue_uav, ue_uav_data_latency)      # 发送数据
        
        # UE-UAV 链路使用 OFDMA：不同 UE 使用不同子载波，新请求与积压并行处理，无需等待

        if self.cache[req_id]:
            # Serve locally: 本地缓存命中
            ue.latency_current_request = ue_uav_transmission_time
        elif self._current_collaborator:
            # === UAV ↔ 协作UAV 链路 ===
            uav_uav_request_latency = request_size_bits / self._uav_uav_rate
            uav_uav_data_latency = file_size_bits / self._uav_uav_rate
            uav_uav_transmission_time = uav_uav_request_latency + uav_uav_data_latency
            
            # 该链路的排队等待时间（全双工 + 本时隙内排队）
            collaborator_id = self._current_collaborator.id
            # 自己作为请求方的等待：backlog + 本时隙已累积的传输时间（全双工取max）
            # 注意：按协作者ID查询，因为不同协作者的积压是独立的
            my_backlog_tx = self._backlog_tx_uav_uav_as_requester.get(collaborator_id, 0.0)
            my_backlog_rx = self._backlog_rx_uav_uav_as_requester.get(collaborator_id, 0.0)
            my_current_tx = self._tx_time_uav_uav_as_requester.get(collaborator_id, 0.0)
            my_current_rx = self._rx_time_uav_uav_as_requester.get(collaborator_id, 0.0)
            self_wait = max(my_backlog_tx + my_current_tx, my_backlog_rx + my_current_rx)
            # 协作者作为协作者角色的等待：使用协作者「作为协作者」时针对我的积压（FDM并行）
            my_tx_at_collaborator = self._current_collaborator._tx_time_uav_uav_as_collaborator.get(self.id, 0.0)
            my_rx_at_collaborator = self._current_collaborator._rx_time_uav_uav_as_collaborator.get(self.id, 0.0)
            my_backlog_tx_at_collaborator = self._current_collaborator._backlog_tx_uav_uav_as_collaborator.get(self.id, 0.0)
            my_backlog_rx_at_collaborator = self._current_collaborator._backlog_rx_uav_uav_as_collaborator.get(self.id, 0.0)
            collaborator_wait = max(my_backlog_tx_at_collaborator + my_tx_at_collaborator,
                                    my_backlog_rx_at_collaborator + my_rx_at_collaborator)
            # 双方都准备好才能开始，取最大值
            queue_wait_uav_uav = max(self_wait, collaborator_wait)
            
            if self._current_collaborator.cache[req_id]:
                # Served by collaborator: 协作UAV缓存命中
                # 自己作为请求方：同一协作者的多个请求串行排队
                if collaborator_id not in self._tx_time_uav_uav_as_requester:
                    self._tx_time_uav_uav_as_requester[collaborator_id] = 0.0
                    self._rx_time_uav_uav_as_requester[collaborator_id] = 0.0
                self._tx_time_uav_uav_as_requester[collaborator_id] += uav_uav_request_latency
                self._rx_time_uav_uav_as_requester[collaborator_id] += uav_uav_data_latency
                # 协作者侧：不同请求方通过FDM并行（带宽已在_set_rates中分割），同一请求方的多个请求串行排队
                if self.id not in self._current_collaborator._rx_time_uav_uav_as_collaborator:
                    self._current_collaborator._rx_time_uav_uav_as_collaborator[self.id] = 0.0
                    self._current_collaborator._tx_time_uav_uav_as_collaborator[self.id] = 0.0
                self._current_collaborator._rx_time_uav_uav_as_collaborator[self.id] += uav_uav_request_latency
                self._current_collaborator._tx_time_uav_uav_as_collaborator[self.id] += uav_uav_data_latency
                
                ue.latency_current_request = (ue_uav_transmission_time +
                                              uav_uav_transmission_time + queue_wait_uav_uav)
            else:
                # Served by MBS through collaborator: 需要从MBS获取
                # === 协作UAV ↔ MBS 链路 ===
                # 上行请求：UAV → MBS，使用 UAV 发射功率
                uav_mbs_request_latency = request_size_bits / self._current_collaborator._uav_mbs_uplink_rate
                # 下行数据：MBS → UAV，使用 MBS 发射功率
                uav_mbs_data_latency = file_size_bits / self._current_collaborator._uav_mbs_downlink_rate
                uav_mbs_transmission_time = uav_mbs_request_latency + uav_mbs_data_latency
                
                # 该链路的排队等待时间（全双工 + 本时隙内排队）
                queue_wait_uav_mbs = max(self._current_collaborator._backlog_tx_uav_mbs + self._current_collaborator._tx_time_uav_mbs,
                                         self._current_collaborator._backlog_rx_uav_mbs + self._current_collaborator._rx_time_uav_mbs)
                
                # 自己作为请求方 UAV-UAV：同一协作者的多个请求串行排队
                if collaborator_id not in self._tx_time_uav_uav_as_requester:
                    self._tx_time_uav_uav_as_requester[collaborator_id] = 0.0
                    self._rx_time_uav_uav_as_requester[collaborator_id] = 0.0
                self._tx_time_uav_uav_as_requester[collaborator_id] += uav_uav_request_latency
                self._rx_time_uav_uav_as_requester[collaborator_id] += uav_uav_data_latency
                # 协作者侧 UAV-UAV：不同请求方通过FDM并行，同一请求方的多个请求串行排队
                if self.id not in self._current_collaborator._rx_time_uav_uav_as_collaborator:
                    self._current_collaborator._rx_time_uav_uav_as_collaborator[self.id] = 0.0
                    self._current_collaborator._tx_time_uav_uav_as_collaborator[self.id] = 0.0
                self._current_collaborator._rx_time_uav_uav_as_collaborator[self.id] += uav_uav_request_latency
                self._current_collaborator._tx_time_uav_uav_as_collaborator[self.id] += uav_uav_data_latency
                # 协作者 UAV-MBS：发送请求 + 接收数据（点对点链路，串行累加）
                self._current_collaborator._tx_time_uav_mbs += uav_mbs_request_latency
                self._current_collaborator._rx_time_uav_mbs += uav_mbs_data_latency
                
                ue.latency_current_request = (ue_uav_transmission_time +
                                              uav_uav_transmission_time + queue_wait_uav_uav +
                                              uav_mbs_transmission_time + queue_wait_uav_mbs)
                _try_add_file_to_cache(self._current_collaborator, req_id)
            _try_add_file_to_cache(self, req_id)
        else:
            # Fetch from MBS via backhaul: 通过回程链路从MBS获取
            # === UAV ↔ MBS 链路 ===
            # 上行请求：UAV → MBS，使用 UAV 发射功率
            uav_mbs_request_latency = request_size_bits / self._uav_mbs_uplink_rate
            # 下行数据：MBS → UAV，使用 MBS 发射功率
            uav_mbs_data_latency = file_size_bits / self._uav_mbs_downlink_rate
            uav_mbs_transmission_time = uav_mbs_request_latency + uav_mbs_data_latency
            
            # 该链路的排队等待时间（全双工 + 本时隙内排队）
            queue_wait_uav_mbs = max(self._backlog_tx_uav_mbs + self._tx_time_uav_mbs,
                                     self._backlog_rx_uav_mbs + self._rx_time_uav_mbs)
            
            # UAV-MBS：发送请求 + 接收数据（点对点链路，串行累加）
            self._tx_time_uav_mbs += uav_mbs_request_latency
            self._rx_time_uav_mbs += uav_mbs_data_latency
            
            ue.latency_current_request = (ue_uav_transmission_time +
                                          uav_mbs_transmission_time + queue_wait_uav_mbs)
            _try_add_file_to_cache(self, req_id)

    def update_ema_and_cache(self) -> None:
        """Update EMA scores and cache reactively."""
        if self.failed:
            return
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores
        self.cache = self._working_cache.copy()  # Update cache after processing all requests of all UAVs

    def gdsf_cache_update(self) -> None:
        """Update cache using the GDSF caching policy at a longer timescale."""
        if self.failed:
            return
        priority_scores = self._ema_scores / config.FILE_SIZES
        sorted_file_ids = np.argsort(-priority_scores)
        self.cache = np.zeros(config.NUM_FILES, dtype=bool)
        used_space = 0.0
        for file_id in sorted_file_ids:
            file_size = config.FILE_SIZES[file_id]
            if used_space + file_size <= config.UAV_STORAGE_CAPACITY[self.id]:
                self.cache[file_id] = True
                used_space += file_size
            else:
                break

    def update_energy_consumption(self) -> None:
        """Update UAV energy consumption for the current time slot.
        
        跨时隙传输模型:
        - 每时隙最多通信 1 秒，超出部分积压到下一时隙
        - 发送和接收分别计算积压
        
        能耗 = 飞行能耗 + 通信能耗
        - 飞行能耗：移动 + 悬停，占满整个时隙
        - 通信能耗：发送功率 × 发送时间 + 接收功率 × 接收时间
        """
        if self.failed and not self._failed_this_step:
            self._energy_current_slot = 0.0
            return
        # 1. 飞行能耗（移动 + 悬停，占满整个时隙）
        # 注：碰撞消解/边界裁剪可能导致实际位移略超出 v*tau，造成 time_hovering 出现极小负数。
        # 这里对时间进行夹紧以保证能耗数值稳定。
        time_moving = self._dist_moved / (config.UAV_SPEED + config.EPSILON)
        time_moving = float(np.clip(time_moving, 0.0, config.TIME_SLOT_DURATION))
        time_hovering = float(np.clip(config.TIME_SLOT_DURATION - time_moving, 0.0, config.TIME_SLOT_DURATION))
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        
        # 2. 通信能耗（分别计算发送和接收，每时隙最多 1 秒）
        # UE-UAV 链路
        total_tx_ue_uav = self._backlog_tx_ue_uav + self._tx_time_ue_uav
        actual_tx_ue_uav = min(total_tx_ue_uav, config.TIME_SLOT_DURATION)
        self._backlog_tx_ue_uav = max(0.0, total_tx_ue_uav - config.TIME_SLOT_DURATION)
        
        total_rx_ue_uav = self._backlog_rx_ue_uav + self._rx_time_ue_uav
        actual_rx_ue_uav = min(total_rx_ue_uav, config.TIME_SLOT_DURATION)
        self._backlog_rx_ue_uav = max(0.0, total_rx_ue_uav - config.TIME_SLOT_DURATION)
        
        # UAV-UAV 链路 - 作为请求方（按协作者ID分组处理积压）
        # 注意：作为请求方时，不同协作者的链路是独立的，各自有独立的时隙限制
        energy_tx_requester = 0.0  # 能耗计算用（累加）
        energy_rx_requester = 0.0
        all_collaborator_ids_tx = set(self._backlog_tx_uav_uav_as_requester.keys()) | set(self._tx_time_uav_uav_as_requester.keys())
        for collab_id in all_collaborator_ids_tx:
            backlog_tx = self._backlog_tx_uav_uav_as_requester.get(collab_id, 0.0)
            current_tx = self._tx_time_uav_uav_as_requester.get(collab_id, 0.0)
            total_tx = backlog_tx + current_tx
            actual_tx = min(total_tx, config.TIME_SLOT_DURATION)
            energy_tx_requester += actual_tx
            self._backlog_tx_uav_uav_as_requester[collab_id] = max(0.0, total_tx - config.TIME_SLOT_DURATION)
        
        all_collaborator_ids_rx = set(self._backlog_rx_uav_uav_as_requester.keys()) | set(self._rx_time_uav_uav_as_requester.keys())
        for collab_id in all_collaborator_ids_rx:
            backlog_rx = self._backlog_rx_uav_uav_as_requester.get(collab_id, 0.0)
            current_rx = self._rx_time_uav_uav_as_requester.get(collab_id, 0.0)
            total_rx = backlog_rx + current_rx
            actual_rx = min(total_rx, config.TIME_SLOT_DURATION)
            energy_rx_requester += actual_rx
            self._backlog_rx_uav_uav_as_requester[collab_id] = max(0.0, total_rx - config.TIME_SLOT_DURATION)
        
        # 清理已完成的积压条目
        self._backlog_tx_uav_uav_as_requester = {k: v for k, v in self._backlog_tx_uav_uav_as_requester.items() if v > 0}
        self._backlog_rx_uav_uav_as_requester = {k: v for k, v in self._backlog_rx_uav_uav_as_requester.items() if v > 0}
        
        # UAV-UAV 链路 - 作为协作者（FDM并行，按请求方分组处理积压）
        # 总功率限制模型：FDM 并行传输共享同一功放，能耗时间取 max
        max_tx_collaborator = 0.0  # 能耗计算用（取max）
        max_rx_collaborator = 0.0
        all_requester_ids = set(self._backlog_tx_uav_uav_as_collaborator.keys()) | set(self._tx_time_uav_uav_as_collaborator.keys())
        for req_id in all_requester_ids:
            backlog_tx = self._backlog_tx_uav_uav_as_collaborator.get(req_id, 0.0)
            current_tx = self._tx_time_uav_uav_as_collaborator.get(req_id, 0.0)
            total_tx = backlog_tx + current_tx
            actual_tx = min(total_tx, config.TIME_SLOT_DURATION)
            max_tx_collaborator = max(max_tx_collaborator, actual_tx)
            self._backlog_tx_uav_uav_as_collaborator[req_id] = max(0.0, total_tx - config.TIME_SLOT_DURATION)
        
        all_requester_ids_rx = set(self._backlog_rx_uav_uav_as_collaborator.keys()) | set(self._rx_time_uav_uav_as_collaborator.keys())
        for req_id in all_requester_ids_rx:
            backlog_rx = self._backlog_rx_uav_uav_as_collaborator.get(req_id, 0.0)
            current_rx = self._rx_time_uav_uav_as_collaborator.get(req_id, 0.0)
            total_rx = backlog_rx + current_rx
            actual_rx = min(total_rx, config.TIME_SLOT_DURATION)
            max_rx_collaborator = max(max_rx_collaborator, actual_rx)
            self._backlog_rx_uav_uav_as_collaborator[req_id] = max(0.0, total_rx - config.TIME_SLOT_DURATION)
        
        # 清理已完成的积压条目（积压为0的可以移除）
        self._backlog_tx_uav_uav_as_collaborator = {k: v for k, v in self._backlog_tx_uav_uav_as_collaborator.items() if v > 0}
        self._backlog_rx_uav_uav_as_collaborator = {k: v for k, v in self._backlog_rx_uav_uav_as_collaborator.items() if v > 0}
        
        # UAV-UAV 能耗时间：
        # - 作为请求方：通常只有一个协作者，直接累加
        # - 作为协作者：FDM 并行共享功放，取 max
        energy_tx_uav_uav = energy_tx_requester + max_tx_collaborator
        energy_rx_uav_uav = energy_rx_requester + max_rx_collaborator
        
        # UAV-MBS 链路
        total_tx_uav_mbs = self._backlog_tx_uav_mbs + self._tx_time_uav_mbs
        actual_tx_uav_mbs = min(total_tx_uav_mbs, config.TIME_SLOT_DURATION)
        self._backlog_tx_uav_mbs = max(0.0, total_tx_uav_mbs - config.TIME_SLOT_DURATION)
        
        total_rx_uav_mbs = self._backlog_rx_uav_mbs + self._rx_time_uav_mbs
        actual_rx_uav_mbs = min(total_rx_uav_mbs, config.TIME_SLOT_DURATION)
        self._backlog_rx_uav_mbs = max(0.0, total_rx_uav_mbs - config.TIME_SLOT_DURATION)
        
        # 通信能耗计算
        # 假设：不同频段（UE-UAV、UAV-UAV、UAV-MBS）使用独立的射频前端，各自有独立的功率预算
        # - UE-UAV 链路：使用 BANDWIDTH_EDGE，OFDMA 并行取 max
        # - UAV-UAV 链路：使用 BANDWIDTH_INTER，FDM 并行取 max（协作者侧）
        # - UAV-MBS 链路：使用 BANDWIDTH_BACKHAUL，点对点
        # 不同频段的能耗时间累加（独立射频前端）
        total_tx_time = actual_tx_ue_uav + energy_tx_uav_uav + actual_tx_uav_mbs
        total_rx_time = actual_rx_ue_uav + energy_rx_uav_uav + actual_rx_uav_mbs
        comm_energy = (config.TRANSMIT_POWER * total_tx_time + 
                       config.RECEIVE_POWER * total_rx_time)
        
        self._energy_current_slot = fly_energy + comm_energy
