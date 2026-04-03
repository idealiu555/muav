"""
注意力机制模块：处理可变长度的 UE 列表
使用 Cross-Attention 架构：UAV 状态作为 Query，UE 特征作为 Key/Value
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """正交初始化，改善梯度流"""
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class UEEmbedding(nn.Module):
    """将 UE 原始特征映射到高维 embedding 空间"""

    def __init__(self, num_files: int, embed_dim: int = config.ATTENTION_EMBED_DIM):
        super().__init__()
        # 根据 embed_dim 动态分配各部分维度
        # 比例: pos : file : cache_hit = 4 : 2 : 2
        pos_dim = embed_dim // 2       # 64 for embed_dim=128
        file_dim = embed_dim // 4      # 32 for embed_dim=128
        cache_dim = embed_dim // 4     # 32 for embed_dim=128

        # 位置特征 embedding (3D -> pos_dim)
        self.pos_embed = layer_init(nn.Linear(3, pos_dim))
        # 文件 ID embedding (离散 -> file_dim)
        self.file_embed = nn.Embedding(num_files, file_dim)
        # 缓存命中 embedding (1D -> cache_dim)
        self.cache_hit_embed = layer_init(nn.Linear(1, cache_dim))
        # LayerNorm 保证训练稳定性
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim  # pos_dim + file_dim + cache_dim = embed_dim

    def forward(self, ue_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ue_features: [batch, num_ues, 5]
                - [:, :, :3]: 相对位置（归一化）
                - [:, :, 3]: 文件 ID（归一化到 [0, 1)，需要反归一化）
                - [:, :, 4]: 缓存命中标志
        Returns:
            ue_embeddings: [batch, num_ues, embed_dim]
        """
        pos = ue_features[:, :, :3]
        # 关键修复：反归一化 file_id，从 [0, 1) 映射回 [0, NUM_FILES)
        # 使用 round() 四舍五入，避免浮点精度问题导致的截断错误
        file_id = (ue_features[:, :, 3] * config.NUM_FILES).round().long().clamp(0, config.NUM_FILES - 1)
        cache_hit = ue_features[:, :, 4:5]

        pos_emb = F.silu(self.pos_embed(pos))
        file_emb = self.file_embed(file_id)
        hit_emb = F.silu(self.cache_hit_embed(cache_hit))

        ue_emb = torch.cat([pos_emb, file_emb, hit_emb], dim=-1)
        return self.layer_norm(ue_emb)


class NeighborEmbedding(nn.Module):
    """将邻居 UAV 特征映射到高维 embedding 空间（混合方案）"""

    def __init__(self, num_files: int = config.NUM_FILES, embed_dim: int = config.ATTENTION_NEIGHBOR_DIM):
        super().__init__()
        # 邻居特征: pos(3) + cache(NUM_FILES) + immediate_help(1) + complementarity(1) + is_active(1)
        # 分配 embedding 维度：cache 信息最丰富，分配更多维度
        pos_dim = embed_dim // 4       # 16 for embed_dim=64
        cache_dim = embed_dim // 2     # 32 for embed_dim=64
        processed_dim = embed_dim // 4  # 16 for embed_dim=64

        self.pos_embed = layer_init(nn.Linear(3, pos_dim))
        self.cache_embed = layer_init(nn.Linear(num_files, cache_dim))
        self.processed_embed = layer_init(nn.Linear(3, processed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim

    def forward(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: [batch, num_neighbors, NEIGHBOR_STATE_DIM]
                - [:, :, :3]: 相对位置
                - [:, :, 3:3+NUM_FILES]: cache bitmap
                - [:, :, -3:]: immediate_help, complementarity, is_active
        Returns:
            neighbor_embeddings: [batch, num_neighbors, embed_dim]
        """
        pos = neighbor_features[:, :, :3]
        cache = neighbor_features[:, :, 3:3 + config.NUM_FILES]
        processed = neighbor_features[:, :, -3:]

        pos_emb = F.silu(self.pos_embed(pos))
        cache_emb = F.silu(self.cache_embed(cache))
        processed_emb = F.silu(self.processed_embed(processed))

        neighbor_emb = torch.cat([pos_emb, cache_emb, processed_emb], dim=-1)
        return self.layer_norm(neighbor_emb)


class UAVEmbedding(nn.Module):
    """将 UAV 状态映射到高维 embedding 空间"""

    def __init__(self, num_files: int, embed_dim: int = config.ATTENTION_UAV_EMBED_DIM):
        super().__init__()
        pos_dim = embed_dim // 4
        cache_dim = embed_dim // 2
        active_dim = embed_dim - pos_dim - cache_dim
        # 位置 embedding (3D -> pos_dim)
        self.pos_embed = layer_init(nn.Linear(3, pos_dim))
        # 缓存 embedding (NUM_FILES -> cache_dim)
        self.cache_embed = layer_init(nn.Linear(num_files, cache_dim))
        # 活跃标记 embedding (1D -> active_dim)
        self.active_embed = layer_init(nn.Linear(1, active_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim

    def forward(self, uav_pos: torch.Tensor, uav_cache: torch.Tensor, uav_active: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uav_pos: [batch, 3] 归一化的 UAV 位置
            uav_cache: [batch, NUM_FILES] 缓存位图
            uav_active: [batch, 1] 活跃标记
        Returns:
            uav_embedding: [batch, embed_dim]
        """
        pos_emb = F.silu(self.pos_embed(uav_pos))
        cache_emb = F.silu(self.cache_embed(uav_cache))
        active_emb = F.silu(self.active_embed(uav_active))
        uav_emb = torch.cat([pos_emb, cache_emb, active_emb], dim=-1)
        return self.layer_norm(uav_emb)


class CrossAttention(nn.Module):
    """
    交叉注意力模块：UAV 状态作为 Query，UE/Neighbor 特征作为 Key/Value

    Q 来自 UAV 状态，表示"UAV 想要关注什么"
    K, V 来自 UE 特征，表示"每个 UE 的关键信息和内容"
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads
        assert kv_dim % num_heads == 0, "kv_dim must be divisible by num_heads"

        # Q 来自 UAV，K, V 来自 UE/Neighbor
        self.q_proj = layer_init(nn.Linear(query_dim, kv_dim))
        self.k_proj = layer_init(nn.Linear(kv_dim, kv_dim))
        self.v_proj = layer_init(nn.Linear(kv_dim, kv_dim))
        self.out_proj = layer_init(nn.Linear(kv_dim, kv_dim))

        self.scale = self.head_dim ** -0.5
        self.output_dim = kv_dim

    def forward(self, query: torch.Tensor, kv: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            query: [batch, query_dim] UAV embedding
            kv: [batch, seq_len, kv_dim] UE/Neighbor embeddings
            mask: [batch, seq_len] 1=真实数据, 0=padding
        Returns:
            output: [batch, kv_dim] 聚合后的特征
        """
        batch_size, seq_len, _ = kv.shape

        # 生成 Q, K, V
        Q = self.q_proj(query).unsqueeze(1)  # [batch, 1, kv_dim]
        K = self.k_proj(kv)  # [batch, seq_len, kv_dim]
        V = self.v_proj(kv)  # [batch, seq_len, kv_dim]

        # 重塑为多头格式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, num_heads, 1, seq_len]

        # 应用 mask（padding 位置设为 -inf）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 处理全 mask 的情况（避免 nan）
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 加权聚合 V
        output = torch.matmul(attn_weights, V)  # [batch, num_heads, 1, head_dim]

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1)
        return self.out_proj(output)  # [batch, kv_dim]


def parse_observation(obs: torch.Tensor) -> dict:
    """
    解析 flat vector 观测为结构化数据

    观测结构：
    [uav_pos(3), uav_cache(NUM_FILES), uav_is_active(1), neighbor_features(...),
     neighbor_count(1), ue_features(MAX_UES*5), ue_count(1)]
    """
    expected_obs_dim = (config.OWN_STATE_DIM +
                        config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_STATE_DIM +
                        1 +
                        config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM +
                        1)
    if obs.shape[1] != expected_obs_dim:
        raise ValueError(
            f"Unexpected observation dim: got {obs.shape[1]}, expected {expected_obs_dim}. "
            "Please ensure env observation structure and config dimensions are aligned."
        )

    batch_size = obs.shape[0]
    idx = 0

    # UAV 状态
    uav_pos = obs[:, idx:idx + 3]
    idx += 3
    uav_cache = obs[:, idx:idx + config.NUM_FILES]
    idx += config.NUM_FILES
    uav_active = obs[:, idx:idx + 1]
    idx += 1

    # 邻居特征
    neighbor_flat = obs[:, idx:idx + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_STATE_DIM]
    neighbor_features = neighbor_flat.view(batch_size, config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_STATE_DIM)
    idx += config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_STATE_DIM

    # 邻居数量
    neighbor_count = obs[:, idx:idx + 1].squeeze(-1)
    idx += 1

    # UE 特征
    ue_flat = obs[:, idx:idx + config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM]
    ue_features = ue_flat.view(batch_size, config.MAX_ASSOCIATED_UES, config.UE_STATE_DIM)
    idx += config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM

    # UE 数量
    ue_count = obs[:, idx:idx + 1].squeeze(-1)

    # 生成 mask
    neighbor_mask = torch.arange(config.MAX_UAV_NEIGHBORS, device=obs.device).expand(batch_size, -1)
    neighbor_mask = (neighbor_mask < neighbor_count.unsqueeze(-1)).float()

    ue_mask = torch.arange(config.MAX_ASSOCIATED_UES, device=obs.device).expand(batch_size, -1)
    ue_mask = (ue_mask < ue_count.unsqueeze(-1)).float()

    return {
        'uav_pos': uav_pos,
        'uav_cache': uav_cache,
        'uav_active': uav_active,
        'neighbor_features': neighbor_features,
        'neighbor_mask': neighbor_mask,
        'ue_features': ue_features,
        'ue_mask': ue_mask,
    }


class AttentionEncoder(nn.Module):
    """
    完整的注意力编码器：将 flat vector 观测编码为固定维度特征

    架构：
    1. 解析观测 -> 结构化数据
    2. UAV Embedding -> [batch, 64]
    3. UE Embedding + CrossAttention -> [batch, 128]
    4. Neighbor Embedding + CrossAttention -> [batch, 64]
    5. 拼接 -> [batch, 256]

    设计原则：
    - 输入维度 380，输出维度 256（轻微压缩）
    - UE attention: heads=2, head_dim=64（符合最佳实践）
    - Neighbor attention: heads=2, head_dim=32（符合最佳实践）
    - UE 维度较大（50个UE，信息量最大）
    """

    def __init__(self, num_files: int = config.NUM_FILES,
                 ue_embed_dim: int = config.ATTENTION_EMBED_DIM,
                 uav_embed_dim: int = config.ATTENTION_UAV_EMBED_DIM,
                 neighbor_out_dim: int = config.ATTENTION_NEIGHBOR_DIM,
                 num_heads: int = config.ATTENTION_NUM_HEADS):
        super().__init__()

        # Embedding 模块
        self.uav_embed = UAVEmbedding(num_files, uav_embed_dim)
        self.ue_embed = UEEmbedding(num_files, ue_embed_dim)
        self.neighbor_embed = NeighborEmbedding(num_files, neighbor_out_dim)

        # 注意力模块
        # UE attention: query=64, kv=128, heads=2 -> head_dim=64 ✓
        self.ue_attention = CrossAttention(uav_embed_dim, ue_embed_dim, num_heads)
        # Neighbor attention: query=64, kv=64, heads=2 -> head_dim=32 ✓
        self.neighbor_attention = CrossAttention(uav_embed_dim, neighbor_out_dim, num_heads)

        # 输出维度：uav(64) + ue_attn(128) + neighbor_attn(64) = 256
        self.output_dim = uav_embed_dim + ue_embed_dim + neighbor_out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim] flat vector 观测
        Returns:
            encoded: [batch, output_dim] 编码后的特征
        """
        # 解析观测
        parsed = parse_observation(obs)

        # UAV embedding
        uav_emb = self.uav_embed(parsed['uav_pos'], parsed['uav_cache'], parsed['uav_active'])

        # UE attention
        ue_emb = self.ue_embed(parsed['ue_features'])
        ue_attn = self.ue_attention(uav_emb, ue_emb, parsed['ue_mask'])

        # Neighbor attention
        neighbor_emb = self.neighbor_embed(parsed['neighbor_features'])
        neighbor_attn = self.neighbor_attention(uav_emb, neighbor_emb, parsed['neighbor_mask'])

        # 拼接所有特征
        return torch.cat([uav_emb, ue_attn, neighbor_attn], dim=-1)


class MeanPoolingEncoder(nn.Module):
    """
    无 attention 分支的轻量编码器。

    使用 count/mask 对邻居和 UE 特征做均值池化（统一权重），
    避免 padding 0 对 MLP 输入造成语义污染。
    """

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = config.OWN_STATE_DIM + config.NEIGHBOR_STATE_DIM + config.UE_STATE_DIM + 2

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        parsed = parse_observation(obs)

        own_state = torch.cat(
            [parsed["uav_pos"], parsed["uav_cache"], parsed["uav_active"]],
            dim=-1,
        )

        neighbor_mask = parsed["neighbor_mask"].unsqueeze(-1)
        neighbor_sum = (parsed["neighbor_features"] * neighbor_mask).sum(dim=1)
        neighbor_count = neighbor_mask.sum(dim=1)
        neighbor_denom = neighbor_count.clamp_min(1.0)
        neighbor_mean = neighbor_sum / neighbor_denom

        ue_mask = parsed["ue_mask"].unsqueeze(-1)
        ue_sum = (parsed["ue_features"] * ue_mask).sum(dim=1)
        ue_count = ue_mask.sum(dim=1)
        ue_denom = ue_count.clamp_min(1.0)
        ue_mean = ue_sum / ue_denom

        neighbor_count_norm = neighbor_count / float(config.MAX_UAV_NEIGHBORS)
        ue_count_norm = ue_count / float(config.MAX_ASSOCIATED_UES)

        return torch.cat(
            [own_state, neighbor_mean, ue_mean, neighbor_count_norm, ue_count_norm],
            dim=-1,
        )


class AgentPoolingAttention(nn.Module):
    """
    Agent 级别的 Permutation-Invariant 聚合模块
    
    将 N 个 agent 的特征聚合为固定维度输出，与 agent 数量无关。
    使用 Self-Attention + Mean Pooling 实现置换不变性。
    
    架构：
    1. 动作嵌入：[batch, N, action_dim] → [batch, N, action_embed_dim]
    2. 特征拼接：[batch, N, encoder_dim + action_embed_dim]
    3. Self-Attention：agents 之间信息交互
    4. Mean Pooling：聚合为固定维度 [batch, output_dim]
    
    优势：
    - Permutation-Invariant：对 agent 顺序不敏感
    - 可扩展：输出维度与 N 无关，支持任意 agent 数量
    """
    
    def __init__(self, encoder_dim: int = 256, action_dim: int = config.ACTION_DIM,
                 action_embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.action_dim = action_dim
        
        # 动作嵌入层
        self.action_embed = nn.Sequential(
            layer_init(nn.Linear(action_dim, action_embed_dim)),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU()
        )
        
        # Agent 特征维度 = encoder_dim + action_embed_dim
        agent_feature_dim = encoder_dim + action_embed_dim
        self.agent_feature_dim = agent_feature_dim
        
        # Self-Attention 层（agents 之间交互）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=agent_feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(agent_feature_dim)
        
        # FFN 层（增强表达能力）
        self.ffn = nn.Sequential(
            layer_init(nn.Linear(agent_feature_dim, agent_feature_dim * 2)),
            nn.SiLU(),
            layer_init(nn.Linear(agent_feature_dim * 2, agent_feature_dim))
        )
        self.ffn_norm = nn.LayerNorm(agent_feature_dim)
        
        # 输出维度与输入 encoder_dim 一致，便于与其他模块兼容
        self.output_proj = layer_init(nn.Linear(agent_feature_dim, encoder_dim))
        self.output_dim = encoder_dim
    
    def forward(
        self,
        agent_encodings: torch.Tensor,
        agent_actions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            agent_encodings: [batch, N, encoder_dim] 所有 agent 的观测编码
            agent_actions: [batch, N, action_dim] 所有 agent 的动作
            active_mask: [batch, N] 1=True=active, 0=False=inactive
        
        Returns:
            pooled: [batch, output_dim] 聚合后的全局特征（与 N 无关）
        """
        batch_size, num_agents, _ = agent_encodings.shape
        if agent_actions.shape[:2] != (batch_size, num_agents):
            raise ValueError(
                f"Expected agent_actions leading shape {(batch_size, num_agents)}, "
                f"got {tuple(agent_actions.shape[:2])}"
            )
        action_emb = self.action_embed(agent_actions)
        agent_features = torch.cat([agent_encodings, action_emb], dim=-1)
        return _pool_active_agent_tokens(
            agent_features=agent_features,
            active_mask=active_mask,
            self_attn=self.self_attn,
            attn_norm=self.attn_norm,
            ffn=self.ffn,
            ffn_norm=self.ffn_norm,
            output_proj=self.output_proj,
        )


class AgentPoolingValue(nn.Module):
    """
    Agent 级别的 permutation-invariant value 聚合模块。

    该模块只聚合 agent 编码，不接收动作输入，适用于 centralized value 估计。
    """

    def __init__(self, encoder_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        if encoder_dim % num_heads != 0:
            raise ValueError("encoder_dim must be divisible by num_heads")

        self.encoder_dim = encoder_dim
        self.self_attn = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.ffn = nn.Sequential(
            layer_init(nn.Linear(encoder_dim, encoder_dim * 2)),
            nn.SiLU(),
            layer_init(nn.Linear(encoder_dim * 2, encoder_dim)),
        )
        self.ffn_norm = nn.LayerNorm(encoder_dim)
        self.output_proj = layer_init(nn.Linear(encoder_dim, encoder_dim))
        self.output_dim = encoder_dim

    def forward(self, agent_encodings: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, encoding_dim = agent_encodings.shape
        if encoding_dim != self.encoder_dim:
            raise ValueError(
                f"Expected agent_encodings last dim {self.encoder_dim}, got {encoding_dim}"
            )
        return _pool_active_agent_tokens(
            agent_features=agent_encodings,
            active_mask=active_mask,
            self_attn=self.self_attn,
            attn_norm=self.attn_norm,
            ffn=self.ffn,
            ffn_norm=self.ffn_norm,
            output_proj=self.output_proj,
        )


def _pool_active_agent_tokens(
    *,
    agent_features: torch.Tensor,
    active_mask: torch.Tensor,
    self_attn: nn.MultiheadAttention,
    attn_norm: nn.LayerNorm,
    ffn: nn.Sequential,
    ffn_norm: nn.LayerNorm,
    output_proj: nn.Linear,
) -> torch.Tensor:
    batch_size, num_agents, _ = agent_features.shape
    if active_mask.shape != (batch_size, num_agents):
        raise ValueError(f"Expected active_mask shape {(batch_size, num_agents)}, got {tuple(active_mask.shape)}")
    if active_mask.dtype != torch.bool:
        raise TypeError(f"active_mask must use dtype torch.bool, got {active_mask.dtype}")

    active_mask = active_mask.to(device=agent_features.device)
    pooled_outputs = agent_features.new_zeros((batch_size, output_proj.out_features))
    valid_rows = active_mask.any(dim=1)
    if not valid_rows.any():
        return pooled_outputs

    features_valid = agent_features[valid_rows]
    mask_valid = active_mask[valid_rows]
    token_mask = mask_valid.unsqueeze(-1).to(dtype=features_valid.dtype)
    key_padding_mask = ~mask_valid

    features_valid = features_valid * token_mask

    attn_out, _ = self_attn(
        features_valid,
        features_valid,
        features_valid,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    features_valid = attn_norm(features_valid + attn_out) * token_mask

    ffn_out = ffn(features_valid)
    features_valid = ffn_norm(features_valid + ffn_out) * token_mask

    pooled_outputs[valid_rows] = output_proj(
        (features_valid * mask_valid.unsqueeze(-1).to(dtype=features_valid.dtype)).sum(dim=1)
        / mask_valid.sum(dim=1, keepdim=True).clamp_min(1.0).to(dtype=features_valid.dtype)
    )
    return pooled_outputs
