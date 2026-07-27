from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        update, _ = self.self_attn(query, query, query, need_weights=False)
        query = self.norms[0](query + update)
        update, _ = self.cross_attn(query, memory, memory, need_weights=False)
        query = self.norms[1](query + update)
        return self.norms[2](query + self.ffn(query))


class LaneDETR(nn.Module):
    """Compact matched DETR/TopoLogic-style centerline decoder with GAL/GCL hooks."""

    def __init__(self, config: dict, feature_channels: int = 512, feature_hw: tuple[int, int] = (4, 4)):
        super().__init__()
        hidden = config["hidden_dim"]
        self.config = config
        self.num_queries = config["num_queries"]
        self.num_points = config["num_points"]
        self.feature_hw = feature_hw
        self.input_proj = nn.Linear(feature_channels, hidden)
        self.query = nn.Embedding(self.num_queries, hidden)
        self.camera_embed = nn.Embedding(7, hidden)
        self.spatial_embed = nn.Embedding(7 * feature_hw[0] * feature_hw[1], hidden)
        self.layers = nn.ModuleList([LaneDecoderLayer(hidden) for _ in range(3)])
        self.class_head = nn.Linear(hidden, 1)
        self.point_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, self.num_points * 2)
        )
        if config["use_gal"]:
            self.prior_semantic = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.prior_position = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.prior_attention = nn.MultiheadAttention(hidden, 8, batch_first=True)
            self.prior_norm = nn.LayerNorm(hidden)

    def _memory(self, features: torch.Tensor, missing_camera: int | None) -> torch.Tensor:
        batch, cameras, channels, height, width = features.shape
        tokens = features.permute(0, 1, 3, 4, 2).reshape(batch, cameras * height * width, channels)
        tokens = self.input_proj(tokens)
        camera_ids = torch.arange(cameras, device=features.device).repeat_interleave(height * width)
        position_ids = torch.arange(cameras * height * width, device=features.device)
        tokens = tokens + self.camera_embed(camera_ids)[None] + self.spatial_embed(position_ids)[None]
        if missing_camera is not None:
            tokens = tokens.clone()
            start = missing_camera * height * width
            tokens[:, start : start + height * width] = 0
        return tokens

    def _gal(
        self, query: torch.Tensor, priors: torch.Tensor, missing_camera: int | None
    ) -> torch.Tensor:
        if missing_camera is not None:
            priors = priors.clone()
            priors[:, missing_camera] = 0
        fused = priors.float().amax(dim=1)
        batch, height, width = fused.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=query.device),
            torch.linspace(0, 1, width, device=query.device),
            indexing="ij",
        )
        positions = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        occupancy = fused.reshape(batch, -1, 1)
        prior_tokens = self.prior_semantic(occupancy) + self.prior_position(positions)[None]
        reference = self.point_head(query).sigmoid().reshape(
            batch, self.num_queries, self.num_points, 2
        ).mean(dim=2)
        distance = torch.cdist(reference, positions[None].expand(batch, -1, -1))
        # Paper's prior mask: -D/k. PyTorch expects (B * heads, Q, K).
        mask = (-distance / self.config["gal_k"]).repeat_interleave(8, dim=0)
        update, _ = self.prior_attention(
            query, prior_tokens, prior_tokens, attn_mask=mask, need_weights=False
        )
        return self.prior_norm(query + update)

    def forward(
        self, features: torch.Tensor, priors: torch.Tensor | None = None, missing_camera: int | None = None
    ) -> dict[str, torch.Tensor]:
        memory = self._memory(features, missing_camera)
        query = self.query.weight[None].expand(features.shape[0], -1, -1)
        for layer_index, layer in enumerate(self.layers):
            query = layer(query, memory)
            if layer_index == 0 and self.config["use_gal"]:
                query = self._gal(query, priors, missing_camera)
        return {
            "logits": self.class_head(query).squeeze(-1),
            "points": self.point_head(query).sigmoid().reshape(
                features.shape[0], self.num_queries, self.num_points, 2
            ),
            "embeddings": query,
        }


def supervised_contrastive(
    embeddings: list[torch.Tensor], groups: list[int], temperature: float = 0.1
) -> torch.Tensor:
    if len(embeddings) < 3:
        return embeddings[0].sum() * 0 if embeddings else torch.tensor(0.0)
    z = F.normalize(torch.stack(embeddings), dim=-1)
    labels = torch.tensor(groups, device=z.device)
    similarity = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None]) & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return z.sum() * 0
    logits = similarity.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()
