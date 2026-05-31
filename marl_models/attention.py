"""
注意力编码模块：处理可变长度的 UE / Neighbor 列表。
使用 UAV embedding 作为条件，通过 query projection + 多层 cross-attention refinement
迭代汇总实体信息；空分支保持为中性 summary。
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
        if embed_dim <= 0 or embed_dim % 4 != 0:
            raise ValueError(f"UEEmbedding embed_dim must be a positive multiple of 4, got {embed_dim}")
        # 根据 embed_dim 动态分配各部分维度
        # 比例: pos : file : cache_hit = 4 : 2 : 2
        pos_dim = embed_dim // 2
        file_dim = embed_dim // 4
        cache_dim = embed_dim // 4

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
        if embed_dim <= 0 or embed_dim % 4 != 0:
            raise ValueError(f"NeighborEmbedding embed_dim must be a positive multiple of 4, got {embed_dim}")
        # 邻居特征: pos(3) + cache(NUM_FILES) + immediate_help(1) + complementarity(1) + is_active(1)
        # 分配 embedding 维度：cache 信息最丰富，分配更多维度
        pos_dim = embed_dim // 4
        cache_dim = embed_dim // 2
        processed_dim = embed_dim // 4

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
    """将 UAV 状态（含实体计数）映射到高维 embedding 空间"""

    def __init__(self, num_files: int, embed_dim: int = config.ATTENTION_UAV_EMBED_DIM):
        super().__init__()
        pos_dim = embed_dim // 4
        cache_dim = embed_dim // 2
        count_dim = config.ATTENTION_COUNT_EMBED_DIM
        active_dim = embed_dim - pos_dim - cache_dim - count_dim
        if active_dim <= 0:
            raise ValueError(
                f"UAVEmbedding embed_dim={embed_dim} is too small for "
                f"pos_dim={pos_dim} + cache_dim={cache_dim} + count_dim={count_dim}"
            )
        self.pos_embed = layer_init(nn.Linear(3, pos_dim))
        self.cache_embed = layer_init(nn.Linear(num_files, cache_dim))
        self.active_embed = layer_init(nn.Linear(1, active_dim))
        self.count_embed = layer_init(nn.Linear(2, count_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim

    def forward(
        self,
        uav_pos: torch.Tensor,
        uav_cache: torch.Tensor,
        uav_active: torch.Tensor,
        normalized_counts: torch.Tensor,
    ) -> torch.Tensor:
        pos_emb = F.silu(self.pos_embed(uav_pos))
        cache_emb = F.silu(self.cache_embed(uav_cache))
        active_emb = F.silu(self.active_embed(uav_active))
        count_emb = F.silu(self.count_embed(normalized_counts))
        uav_emb = torch.cat([pos_emb, cache_emb, active_emb, count_emb], dim=-1)
        return self.layer_norm(uav_emb)


class CrossAttention(nn.Module):
    """
    交叉注意力模块：分支 summary/query 向量关注实体 token 序列。

    Q 表示当前分支的查询或摘要向量。
    K, V 来自通用实体 token 序列（UE 或 Neighbor）。
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if kv_dim % num_heads != 0:
            raise ValueError(f"kv_dim ({kv_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads

        # Q 来自当前分支的 query/summary，K/V 来自实体 token 序列
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
            query: [batch, query_dim] 分支 query / summary 向量
            kv: [batch, seq_len, kv_dim] 实体 token 序列（UE 或 Neighbor）
            mask: [batch, seq_len] 1=有效实体 token, 0=padding
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


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(dim, dim * 4))
        self.fc2 = layer_init(nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class AttentionBlock(nn.Module):
    """Pre-LN residual cross-attention block: norm before each sublayer."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = CrossAttention(dim, dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_norm = self.norm1(query)
        attended = self.attn(q_norm, kv, mask)
        x = query + attended
        z = self.norm2(x)
        x = x + self.ffn(z)
        return x


def zero_empty_summary(summary: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """对空实体分支输出中性 summary，避免残差 query 泄露到结果中。"""
    if mask is None:
        return summary
    has_tokens = mask.sum(dim=1, keepdim=True) > 0
    return summary * has_tokens.to(summary.dtype)


def build_attention_stack(dim: int, num_heads: int, num_layers: int) -> nn.ModuleList:
    """Create a residual cross-attention stack with unique block instances."""
    return nn.ModuleList([AttentionBlock(dim, num_heads) for _ in range(num_layers)])


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
    neighbor_count = obs[:, idx:idx + 1].squeeze(-1).round().clamp(0, config.MAX_UAV_NEIGHBORS)
    idx += 1

    # UE 特征
    ue_flat = obs[:, idx:idx + config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM]
    ue_features = ue_flat.view(batch_size, config.MAX_ASSOCIATED_UES, config.UE_STATE_DIM)
    idx += config.MAX_ASSOCIATED_UES * config.UE_STATE_DIM

    # UE 数量
    ue_count = obs[:, idx:idx + 1].squeeze(-1).round().clamp(0, config.MAX_ASSOCIATED_UES)

    # 生成 mask
    neighbor_mask = torch.arange(config.MAX_UAV_NEIGHBORS, device=obs.device).expand(batch_size, -1)
    neighbor_slot_mask = neighbor_mask < neighbor_count.unsqueeze(-1)
    neighbor_active_mask = neighbor_features[:, :, -1] > 0.5
    neighbor_mask = (neighbor_slot_mask & neighbor_active_mask).float()

    ue_mask = torch.arange(config.MAX_ASSOCIATED_UES, device=obs.device).expand(batch_size, -1)
    ue_mask = (ue_mask < ue_count.unsqueeze(-1)).float()

    return {
        'uav_pos': uav_pos,
        'uav_cache': uav_cache,
        'uav_active': uav_active,
        'neighbor_features': neighbor_features,
        'neighbor_count': neighbor_count,
        'neighbor_mask': neighbor_mask,
        'ue_features': ue_features,
        'ue_count': ue_count,
        'ue_mask': ue_mask,
    }


class AttentionEncoder(nn.Module):
    """Encodes a flat observation into a fixed-dimension feature vector via
    structured cross-attention over UAV, UE, and Neighbor entities.

    Architecture:
    1. Parse observation → structured tensors
    2. UAV Embedding → [batch, ATTENTION_UAV_EMBED_DIM]
    3. UAV summary query projected to per-branch dimensions
    4. UE / Neighbor Embedding + stacked residual CrossAttention
    5. Empty entity branches explicitly zeroed
    6. Concatenate → [batch, output_dim]
    """

    def __init__(self, num_files: int = config.NUM_FILES,
                 ue_embed_dim: int = config.ATTENTION_EMBED_DIM,
                 uav_embed_dim: int = config.ATTENTION_UAV_EMBED_DIM,
                 neighbor_out_dim: int = config.ATTENTION_NEIGHBOR_DIM,
                 num_heads: int = config.ATTENTION_NUM_HEADS,
                 num_layers: int = config.ATTENTION_NUM_LAYERS):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"AttentionEncoder requires num_layers > 0, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"AttentionEncoder num_heads must be positive, got {num_heads}")
        if ue_embed_dim <= 0 or ue_embed_dim % 4 != 0 or ue_embed_dim % num_heads != 0:
            raise ValueError(
                "AttentionEncoder ue_embed_dim must be a positive multiple of 4 "
                f"and divisible by num_heads, got ue_embed_dim={ue_embed_dim}, num_heads={num_heads}"
            )
        if neighbor_out_dim <= 0 or neighbor_out_dim % 4 != 0 or neighbor_out_dim % num_heads != 0:
            raise ValueError(
                "AttentionEncoder neighbor_out_dim must be a positive multiple of 4 "
                f"and divisible by num_heads, got neighbor_out_dim={neighbor_out_dim}, num_heads={num_heads}"
            )
        pos_dim = uav_embed_dim // 4
        cache_dim = uav_embed_dim // 2
        if uav_embed_dim - pos_dim - cache_dim - config.ATTENTION_COUNT_EMBED_DIM <= 0:
            raise ValueError(f"AttentionEncoder uav_embed_dim is too small, got {uav_embed_dim}")

        self.uav_embed = UAVEmbedding(num_files, uav_embed_dim)
        self.ue_embed = UEEmbedding(num_files, ue_embed_dim)
        self.neighbor_embed = NeighborEmbedding(num_files, neighbor_out_dim)

        self.ue_query_proj = layer_init(nn.Linear(uav_embed_dim, ue_embed_dim))
        self.neighbor_query_proj = layer_init(nn.Linear(uav_embed_dim, neighbor_out_dim))
        self.ue_attention_blocks = build_attention_stack(ue_embed_dim, num_heads, num_layers)
        self.neighbor_attention_blocks = build_attention_stack(neighbor_out_dim, num_heads, num_layers)

        self.output_dim = uav_embed_dim + ue_embed_dim + neighbor_out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim] flat vector observation
        Returns:
            encoded: [batch, output_dim]
        """
        parsed = parse_observation(obs)

        normalized_counts = torch.stack(
            [
                (parsed['neighbor_count'] / config.MAX_UAV_NEIGHBORS).clamp(0.0, 1.0),
                (parsed['ue_count'] / config.MAX_ASSOCIATED_UES).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        uav_emb = self.uav_embed(parsed['uav_pos'], parsed['uav_cache'], parsed['uav_active'], normalized_counts)

        ue_emb = self.ue_embed(parsed['ue_features'])
        ue_attn = self.ue_query_proj(uav_emb)
        for block in self.ue_attention_blocks:
            ue_attn = block(ue_attn, ue_emb, parsed['ue_mask'])
        ue_attn = zero_empty_summary(ue_attn, parsed['ue_mask'])

        neighbor_emb = self.neighbor_embed(parsed['neighbor_features'])
        neighbor_attn = self.neighbor_query_proj(uav_emb)
        for block in self.neighbor_attention_blocks:
            neighbor_attn = block(neighbor_attn, neighbor_emb, parsed['neighbor_mask'])
        neighbor_attn = zero_empty_summary(neighbor_attn, parsed['neighbor_mask'])

        return torch.cat([uav_emb, ue_attn, neighbor_attn], dim=-1)
