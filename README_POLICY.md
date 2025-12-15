# Learned Policy Guidance for MAPF

This branch implements learned neural network policies to guide Conflict-Based Search (CBS) by reordering successor nodes during low-level A* search.

## Overview

The policy approach uses a **GNN → Transformer** architecture to score successor moves, helping A* explore more promising paths first while preserving optimality. The policy is trained on optimal A* paths via supervised learning and can reduce search effort on typical instances, though it adds inference overhead.

### Architecture

- **GNN (Graph Neural Network)**: Encodes map structure, agent state (position, goal, timestep), and successor features into 32-dimensional embeddings
- **Lightweight Transformer**: Applies self-attention to produce 64-dimensional hidden representations, then projects to scalar scores for each successor
- **Limitations**: Lacks encoders specifically designed for pathfinding problems; limited long-term memory due to lightweight attention mechanism

### Key Features

- **Optimality-preserving**: Reorders successors without pruning, maintaining solution quality
- **Map-specific training**: One model per map, enabling efficient inference via cached map encoding
- **Per-instance policy grids**: Optional pre-computed grid scores for faster heuristic augmentation

## Files

### Core Implementation
- `policy_guidance.py` - Policy model architecture (MapEncoder CNN + AgentEncoder MLP + PolicyHead)
- `train_gnn_policy.py` - Training script for learning policies from optimal paths
- `policy_grid.py` - Utilities for loading per-cell policy score grids
- `single_agent_planner.py` - A* with policy guidance integration
- `cbs.py` - CBS solver with policy-guided low-level search

### Checkpoints
- `checkpoints/map_policy.pt` - Trained policy checkpoint for the map
- `checkpoints/test_*_policy_grid.csv` - Per-instance policy grid scores (auto-loaded)

### Results
- `results_policy.csv` - Experimental results with policy guidance

## Training a Policy

Train a policy for a specific map instance:

```bash
python train_gnn_policy.py \
  --instance instances/test_1.txt \
  --episodes 200 \
  --batch-size 256 \
  --lr 1e-3 \
  --dropout 0.1 \
  --device auto \
  --save-path checkpoints/map_policy.pt \
  --export-grid checkpoints/test_1_policy_grid.csv
```

Train policies for all test instances:

```bash
python train_gnn_policy.py \
  --instance "instances/test_*.txt" \
  --episodes 200 \
  --batch-size 256 \
  --lr 1e-3 \
  --dropout 0.1 \
  --device auto
```

### Training Parameters

- `--episodes`: Number of training episodes (default: 200)
- `--batch-size`: Transitions per optimization step (default: 256)
- `--lr`: Learning rate for Adam optimizer (default: 1e-3)
- `--dropout`: Dropout rate for regularization (default: 0.1)
- `--device`: Device selection (auto, cuda, mps, cpu)

### How Training Works

1. **Data Generation**: For each episode, sample random start-goal pairs on the map and compute optimal A* paths
2. **Supervision**: Each transition (state, action, next_state) becomes a training sample where the target action is the one chosen by the optimal path
3. **Loss**: Cross-entropy loss maximizes the score of optimal successors relative to other valid moves
4. **Optimization**: Adam optimizer with gradient clipping (max norm 1.0)

## Running Experiments with Policy

### Single Instance

```bash
python run_experiments.py \
  --instance instances/test_1.txt \
  --solver CBS \
  --disjoint \
  --policy-checkpoint checkpoints/map_policy.pt \
  --auto-policy-grid
```

### Batch Experiments (All Test Cases)

```bash
python run_experiments.py \
  --instance "instances/test_*.txt" \
  --solver CBS \
  --batch \
  --disjoint \
  --policy-checkpoint checkpoints/map_policy.pt \
  --auto-policy-grid
```

### Policy Options

- `--policy-checkpoint PATH`: Path to trained policy checkpoint (e.g., `checkpoints/map_policy.pt`)
- `--auto-policy-grid`: Automatically load per-instance policy grids from `checkpoints/test_*_policy_grid.csv`
- `--policy-grid PATH`: Manually specify a policy grid CSV file
- `--policy-grid-weight FLOAT`: Weight for policy grid bonus in f-value (default: 0.0)
- `--policy-grid-apply-to-h`: Apply policy weight to heuristic rather than f-value directly

## Results Summary

From experiments on 50 test instances:

### Performance Highlights

| Metric | A* Baseline | Policy | Improvement |
|--------|-------------|--------|-------------|
| **Median Expanded** | 6 | 5 | ✓ Best |
| **Median Generated** | 11 | 8 | ✓ Best |
| **Median CPU Time** | 0.0027s | 0.022s | 8× slower |
| **Mean Expanded** | 15.1 | 72.8 | Worse (test_47 outlier) |
| **Mean CPU Time** | 0.0066s | 1.002s | 152× slower (test_47 outlier) |
| **Solution Quality** | Optimal | Optimal | ✓ Preserved |

### Key Observations

✓ **Reduces search effort on typical instances**: Achieves lowest median node counts among all methods

✗ **High inference overhead**: Neural network forward passes dominate runtime, especially on hard instances

✓ **Preserves optimality**: All 50 instances achieved same cost as baseline

✗ **Struggles on complex instances**: test_47 took 43.18s vs 0.90s for SMA* due to accumulated inference cost

### When to Use Policy Guidance

**Recommended for:**
- Instances where search trees are small-to-medium (< 1000 high-level nodes)
- Scenarios where solution quality is critical (optimality preserved)
- Repeated queries on the same map (amortize training cost)

**Not recommended for:**
- Very hard instances with large search trees (inference overhead dominates)
- Real-time applications requiring low latency
- Single-query scenarios (training overhead too high)

## Model Architecture Details

### Hyperparameters
- Map embedding dimension: 32
- Agent embedding dimension: 32
- Transformer hidden dimension: 64
- Dropout rate: 0.1

### Input Features
- **Map**: Binary grid (obstacle/free)
- **Agent**: Normalized position (x, y), goal (gx, gy), timestep (tanh-scaled)
- **Successors**: Normalized delta-x, delta-y, Manhattan distance (tanh-scaled), blocked flag

### Output
- Scalar score per successor (higher = better)
- Used to reorder children before A* expansion

## Limitations

1. **No pathfinding-specific encoders**: Generic GNN/transformer architecture not specialized for MAPF
2. **Limited long-term memory**: Lightweight attention may not capture complex dependencies in deep search
3. **Inference overhead**: Per-successor neural network forward pass adds significant computation
4. **Map-specific**: Requires training separate model for each unique map layout

## Future Work

- Lightweight policy architectures (pruned networks, knowledge distillation)
- Hybrid approaches: selectively apply policy only when beneficial (e.g., early in search)
- PathFinding-specific encoders: incorporate domain knowledge (corridor detection, bottleneck analysis)
- Long-term memory: recurrent architectures or memory-augmented networks
- Transfer learning: pre-train on diverse maps, fine-tune per instance

## Citation

If you use this policy guidance approach, please cite:

```bibtex
@article{mapf_policy_2025,
  title={Multi-Agent Pathfinding with CBS, Memory-Bounded Search, and Learned Policies},
  author={Li, Eric and Nguyen, Duc Viet and Yang, Ricky},
  year={2025},
  institution={Simon Fraser University}
}
```

## License

See main project README for license information.

