"""
Map-conditioned policy training for MAPF (successor ordering for A* / CBS).

This script targets the realistic regime where the *same map is reused* across
many MAPF queries. We cache a map encoder once, then train a lightweight policy
to score successor moves conditioned on the current agent state and goal.

The learned policy can be plugged in as a tie-breaker / ordering heuristic
without sacrificing optimality (no pruning). See `PolicyGuidance` in
`policy_guidance.py` for inference-time integration.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple
import glob
import os

import torch
import torch.nn.functional as F

from policy_guidance import PolicyModel, PolicyGuidance, map_to_tensor
from run_experiments import import_mapf_instance
from single_agent_planner import a_star, compute_heuristics

Coord = Tuple[int, int]

# Action order matches single_agent_planner.move directions + wait
ACTION_DELTAS: Sequence[Coord] = [
    (0, -1),  # up (y-1)
    (1, 0),   # right (x+1)
    (0, 1),   # down (y+1)
    (-1, 0),  # left (x-1)
    (0, 0),   # wait
]


def collect_free_cells(my_map: List[List[bool]]) -> List[Coord]:
    free = []
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if not my_map[x][y]:
                free.append((x, y))
    return free


def sample_start_goal(free_cells: Sequence[Coord]) -> Tuple[Coord, Coord]:
    start, goal = random.sample(free_cells, 2)
    return start, goal


def build_successors(curr: Coord, my_map: List[List[bool]]) -> Tuple[List[Coord], List[bool]]:
    h, w = len(my_map), len(my_map[0])
    succ, blocked = [], []
    for dx, dy in ACTION_DELTAS:
        nx, ny = curr[0] + dx, curr[1] + dy
        in_bounds = 0 <= nx < h and 0 <= ny < w
        is_blocked = (not in_bounds) or my_map[nx][ny]
        succ.append((nx, ny))
        blocked.append(is_blocked)
    return succ, blocked


def normalize_coord(val: int, limit: int) -> float:
    return (2.0 * val / max(1, limit - 1)) - 1.0


def build_features(
    curr: Coord,
    goal: Coord,
    timestep: int,
    successors: Sequence[Coord],
    blocked: Sequence[bool],
    h: int,
    w: int,
):
    agent_feat = torch.tensor(
        [
            normalize_coord(curr[0], h),
            normalize_coord(curr[1], w),
            normalize_coord(goal[0], h),
            normalize_coord(goal[1], w),
            math.tanh(0.01 * timestep),
        ],
        dtype=torch.float32,
    )

    succ_feats = []
    for (sx, sy), is_blocked in zip(successors, blocked):
        dx = sx - curr[0]
        dy = sy - curr[1]
        manhattan = abs(dx) + abs(dy)
        succ_feats.append(
            [
                normalize_coord(dx, h),
                normalize_coord(dy, w),
                math.tanh(0.1 * manhattan),
                1.0 if is_blocked else 0.0,
            ]
        )
    succ_feats = torch.tensor(succ_feats, dtype=torch.float32)
    return agent_feat, succ_feats


def path_to_training_samples(path: List[Coord]) -> List[Tuple[Coord, Coord, int]]:
    samples = []
    for t in range(len(path) - 1):
        samples.append((path[t], path[t + 1], t))
    return samples


def find_action_index(successors: Sequence[Coord], target: Coord) -> int:
    for idx, loc in enumerate(successors):
        if loc == target:
            return idx
    # Should not happen for valid paths; fallback to wait
    return len(successors) - 1


def train_policy(
    my_map: List[List[bool]],
    episodes: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dropout: float,
):
    h, w = len(my_map), len(my_map[0])
    free_cells = collect_free_cells(my_map)
    map_tensor = map_to_tensor(my_map).to(device)

    model = PolicyModel(dropout=dropout)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    start_time = time.time()
    for ep in range(episodes):
        samples = []
        while len(samples) < batch_size:
            start, goal = sample_start_goal(free_cells)
            heuristics = compute_heuristics(my_map, goal)
            path = a_star(my_map, start, goal, heuristics, agent=0, constraints=[])
            if path is None or len(path) < 2:
                continue
            for (curr, nxt, t) in path_to_training_samples(path):
                samples.append((curr, goal, nxt, t))
                if len(samples) >= batch_size:
                    break

        agent_feats, succ_feats, targets = [], [], []
        for curr, goal, nxt, t in samples:
            successors, blocked = build_successors(curr, my_map)
            agent_feat, s_feats = build_features(curr, goal, t, successors, blocked, h, w)
            target = find_action_index(successors, nxt)
            agent_feats.append(agent_feat)
            succ_feats.append(s_feats)
            targets.append(target)

        agent_tensor = torch.stack(agent_feats, dim=0).to(device)
        succ_tensor = torch.stack(succ_feats, dim=0).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        map_batch = map_tensor.expand(len(samples), -1, -1, -1).contiguous()

        logits = model(map_batch, agent_tensor, succ_tensor)
        loss = F.cross_entropy(logits, target_tensor)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == target_tensor).float().mean().item()
        log.append((loss.item(), acc))

        if (ep + 1) % 10 == 0:
            avg_loss = sum([l for l, _ in log[-10:]]) / 10.0
            avg_acc = sum([a for _, a in log[-10:]]) / 10.0
            elapsed = time.time() - start_time
            print(f"[ep {ep+1:04d}] loss={avg_loss:.4f} acc={avg_acc:.3f} time={elapsed:.1f}s")

    return model


def save_checkpoint(model: PolicyModel, path: Path):
    map_dim = model.map_encoder.net[0].out_channels
    agent_dim = model.agent_encoder.mlp[0].out_features
    hidden_dim = model.head.hidden_dim
    dropout = model.head.fc[2].p if isinstance(model.head.fc[2], torch.nn.Dropout) else 0.0
    ckpt = {
        "model": model.state_dict(),
        "model_kwargs": {"map_dim": map_dim, "agent_dim": agent_dim, "hidden_dim": hidden_dim, "dropout": dropout},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")


def export_policy_grid(my_map: List[List[bool]], model: PolicyModel, device: torch.device, out_path: Path):
    """
    Export a per-cell policy score grid (CSV) by querying the learned policy
    with a zero-motion (wait) successor at each free cell. Obstacles are 0.
    """
    guidance = PolicyGuidance(model=model, map_tensor=map_to_tensor(my_map), device=device)
    h, w = len(my_map), len(my_map[0])
    grid = []
    for x in range(h):
        row = []
        for y in range(w):
            if my_map[x][y]:
                row.append(0.0)
                continue
            score = guidance.score_successors((x, y), (x, y), 0, successors=[(x, y)], blocked=[False])[0]
            row.append(score)
        grid.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in grid:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"Saved policy grid to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a map-conditioned policy to guide CBS/A*.")
    parser.add_argument("--instance", type=str, default="instances/exp1.txt", help="MAPF instance file or glob (map is reused).")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=256, help="Transitions per optimization step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, or cpu.")
    parser.add_argument("--save-path", type=str, default=None, help="Where to store the trained model. If multiple instances, defaults to checkpoints/<basename>_policy.pt")
    parser.add_argument("--export-grid", type=str, default=None, help="Optional: path to save the per-cell policy grid CSV. If multiple instances, defaults to checkpoints/<basename>_policy_grid.csv")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the policy head to reduce overfitting.")
    args = parser.parse_args()

    instance_paths = sorted(glob.glob(args.instance))
    if not instance_paths:
        raise FileNotFoundError(f"No instances matched pattern: {args.instance}")

    def select_device(name: str) -> torch.device:
        if name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    for inst in instance_paths:
        print(f"=== Training on {inst} ===")
        my_map, _, _ = import_mapf_instance(inst)
        device = select_device(args.device)
        if device.type != args.device and args.device != "auto":
            print(f"Warning: requested device {args.device} not available, using {device}")
        model = train_policy(my_map, episodes=args.episodes, batch_size=args.batch_size, lr=args.lr, device=device, dropout=args.dropout)

        base = os.path.splitext(os.path.basename(inst))[0]
        save_path = Path(args.save_path) if args.save_path else Path("checkpoints") / f"{base}_policy.pt"
        save_checkpoint(model, save_path)
        export_path = None
        if args.export_grid is not None:
            export_path = Path(args.export_grid)
        elif args.export_grid is None:
            export_path = Path("checkpoints") / f"{base}_policy_grid.csv"
        if export_path:
            export_policy_grid(my_map, model, device, export_path)


if __name__ == "__main__":
    main()

