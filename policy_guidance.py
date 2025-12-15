import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Typing aliases
Coord = Tuple[int, int]
Successor = Coord


def map_to_tensor(my_map: List[List[bool]]) -> torch.Tensor:
    """Convert a boolean grid into a float tensor shaped (1, 1, H, W)."""
    h, w = len(my_map), len(my_map[0])
    grid = torch.zeros((1, 1, h, w), dtype=torch.float32)
    for i in range(h):
        for j in range(w):
            grid[0, 0, i, j] = 1.0 if my_map[i][j] else 0.0
    return grid


class MapEncoder(nn.Module):
    """
    Lightweight CNN encoder for a static grid map.
    Output:
        spatial_features: (B, C, H, W)
        global_features: (B, C)
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.net(x)
        # Simple global pooling to summarize the map
        global_feat = F.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1)
        return spatial, global_feat


class AgentEncoder(nn.Module):
    """Encode agent-specific tokens (current, goal, timestep)."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PolicyHead(nn.Module):
    """
    Scores successor moves given:
        - global map embedding
        - encoded agent token
        - per-successor features
    """

    def __init__(self, map_dim: int, agent_dim: int, hidden_dim: int = 64, succ_feat_dim: int = 4, dropout: float = 0.0):
        """
        A small MLP over concatenated map+agent+successor features.

        Note: Older checkpoints (e.g., checkpoints/map_policy.pt) were trained
        without a Dropout layer, so we build the sequence conditionally to keep
        state_dict keys compatible (Linear, ReLU, [optional Dropout], Linear).
        """
        super().__init__()
        input_dim = map_dim + agent_dim + succ_feat_dim
        self.hidden_dim = hidden_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, map_global: torch.Tensor, agent_emb: torch.Tensor, succ_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_global: (B, Hm)
            agent_emb: (B, Ha)
            succ_feats: (B, K, 4) with delta-x, delta-y, manhattan, is_blocked
        Returns:
            scores: (B, K)
        """
        bsz, k, _ = succ_feats.shape
        map_expand = map_global.unsqueeze(1).expand(-1, k, -1)
        agent_expand = agent_emb.unsqueeze(1).expand(-1, k, -1)
        x = torch.cat([map_expand, agent_expand, succ_feats], dim=-1)
        scores = self.fc(x).squeeze(-1)
        return scores


class PolicyModel(nn.Module):
    """Map-conditioned policy that reorders successors."""

    def __init__(self, map_dim: int = 32, agent_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.map_encoder = MapEncoder(hidden_dim=map_dim)
        self.agent_encoder = AgentEncoder(hidden_dim=agent_dim)
        self.head = PolicyHead(map_dim=map_dim, agent_dim=agent_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        map_tensor: torch.Tensor,
        agent_features: torch.Tensor,
        successor_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            map_tensor: (B, 1, H, W)
            agent_features: (B, 5) -> [x_norm, y_norm, gx_norm, gy_norm, t_norm]
            successor_features: (B, K, 4) -> [dx_norm, dy_norm, manhattan_norm, blocked]
        Returns:
            scores over successors: (B, K)
        """
        _, map_global = self.map_encoder(map_tensor)
        agent_emb = self.agent_encoder(agent_features)
        return self.head(map_global, agent_emb, successor_features)


@dataclass
class PolicyGuidance:
    """
    Light-weight wrapper to reuse a cached map encoding and score successor moves.
    This can be plugged into A* / CBS as a tie-breaker or ordering heuristic.
    """

    model: PolicyModel
    map_tensor: torch.Tensor
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.model.to(self.device)
        self.map_tensor = self.map_tensor.to(self.device)
        # Cache global map embedding
        with torch.no_grad():
            self._map_spatial, self._map_global = self.model.map_encoder(self.map_tensor)

    def score_successors(
        self,
        curr: Coord,
        goal: Coord,
        timestep: int,
        successors: Sequence[Successor],
        blocked: Sequence[bool],
    ) -> List[float]:
        """
        Returns a score for each successor (higher = better).
        Does not perform pruning; use as ordering to preserve optimality.
        """
        h, w = self.map_tensor.shape[-2:]
        norm = lambda x, denom: (2.0 * x / max(1, denom - 1)) - 1.0  # scale to [-1, 1]
        agent_feat = torch.tensor(
            [
                norm(curr[0], h),
                norm(curr[1], w),
                norm(goal[0], h),
                norm(goal[1], w),
                math.tanh(0.01 * timestep),
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        succ_feats = []
        for (sx, sy), is_blocked in zip(successors, blocked):
            dx = sx - curr[0]
            dy = sy - curr[1]
            manhattan = abs(dx) + abs(dy)
            succ_feats.append(
                [
                    norm(dx, h),
                    norm(dy, w),
                    math.tanh(0.1 * manhattan),
                    1.0 if is_blocked else 0.0,
                ]
            )
        succ_feats = torch.tensor(succ_feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            scores = self.model.head(self._map_global, self.model.agent_encoder(agent_feat), succ_feats)
        return scores.squeeze(0).tolist()

    def reorder_successors(
        self,
        curr: Coord,
        goal: Coord,
        timestep: int,
        successors: Sequence[Successor],
        blocked: Sequence[bool],
    ) -> List[Successor]:
        scores = self.score_successors(curr, goal, timestep, successors, blocked)
        order = sorted(range(len(successors)), key=lambda i: scores[i], reverse=True)
        return [successors[i] for i in order]


def load_policy(checkpoint_path: str, my_map: List[List[bool]], device: str = "cpu") -> PolicyGuidance:
    """
    Restore a PolicyGuidance wrapper from a checkpoint produced by train_gnn_policy.py.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PolicyModel(**ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model"])
    map_tensor = map_to_tensor(my_map)
    return PolicyGuidance(model=model, map_tensor=map_tensor, device=torch.device(device))

