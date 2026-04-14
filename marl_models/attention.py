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
    交叉注意力模块：分支 summary/query 向量关注实体 token 序列。

    Q 表示当前分支的查询或摘要向量。
    K, V 来自通用实体 token 序列（UE 或 Neighbor）。
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = kv_dim // num_heads
        assert kv_dim % num_heads == 0, "kv_dim must be divisible by num_heads"

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
    """残差注意力块：使用当前 summary query 迭代细化实体汇总。"""

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
        attended = self.attn(query, kv, mask)
        x = self.norm1(attended + query)
        x = self.norm2(x + self.ffn(x))
        return x


def zero_empty_summary(summary: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """对空实体分支输出中性 summary，避免残差 query 泄露到结果中。"""
    if mask is None:
        return summary
    has_tokens = mask.sum(dim=1, keepdim=True) > 0
    return summary * has_tokens.to(summary.dtype)


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
    3. UAV summary query 投影到各分支维度
    4. UE / Neighbor Embedding + 两层残差 CrossAttention 迭代细化 summary
    5. 空分支显式归零，保持“无实体摘要”语义
    6. 拼接 -> [batch, 256]

    设计原则：
    - 输入维度 380，输出维度 256（轻微压缩）
    - 每个分支先用 query projection 对齐维度，再用两层 attention block 做摘要 refinement
    - UE 分支输出 128 维，Neighbor 分支输出 64 维，保持外部接口不变
    - 当 UE / Neighbor 数量为 0 时，对应分支输出保持中性 summary
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
        self.ue_query_proj = layer_init(nn.Linear(uav_embed_dim, ue_embed_dim))
        self.neighbor_query_proj = layer_init(nn.Linear(uav_embed_dim, neighbor_out_dim))
        self.ue_attention_blocks = nn.ModuleList([
            AttentionBlock(ue_embed_dim, num_heads),
            AttentionBlock(ue_embed_dim, num_heads),
        ])
        self.neighbor_attention_blocks = nn.ModuleList([
            AttentionBlock(neighbor_out_dim, num_heads),
            AttentionBlock(neighbor_out_dim, num_heads),
        ])

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
        ue_attn = self.ue_query_proj(uav_emb)
        for block in self.ue_attention_blocks:
            ue_attn = block(ue_attn, ue_emb, parsed['ue_mask'])
        ue_attn = zero_empty_summary(ue_attn, parsed['ue_mask'])

        # Neighbor attention
        neighbor_emb = self.neighbor_embed(parsed['neighbor_features'])
        neighbor_attn = self.neighbor_query_proj(uav_emb)
        for block in self.neighbor_attention_blocks:
            neighbor_attn = block(neighbor_attn, neighbor_emb, parsed['neighbor_mask'])
        neighbor_attn = zero_empty_summary(neighbor_attn, parsed['neighbor_mask'])

        # 拼接所有特征
        return torch.cat([uav_emb, ue_attn, neighbor_attn], dim=-1)


