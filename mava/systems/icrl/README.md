# Independent Contrastive Reinforcement Learning (ICRL)

Goal-conditioned multi-agent reinforcement learning system using contrastive learning for distance-based Q-value estimation.

## Overview

ICRL learns goal-conditioned policies by:
- Using contrastive learning (InfoNCE loss) to learn state-action and goal representations
- Computing Q-values as negative distances between representations
- Performing hindsight relabeling to generate synthetic goal-reaching experiences

## Key Components

- **Networks**: Separate encoders for state-action pairs and goals
- **Buffer**: Trajectory-based with hindsight goal relabeling
- **Actor**: Gumbel-Softmax for discrete action spaces
- **Critic**: Contrastive distance metric with temperature parameter

## Configuration

Main parameters in `configs/system/icrl/ff_icrl.yaml`:
- `obs_dim`: Base observation dimensions (without goal)
- `goal_dim`: Goal dimensions appended to observations
- `goal_start_idx`/`goal_end_idx`: Indices for goal extraction during relabeling
- `icrl_alpha`: Contrastive loss temperature parameter

## Requirements

- Custom JaxMARL environment with goal-based observations
- Environment must provide `episode_seed` for correct hindsight relabeling
- Observations format: `[state (obs_dim), goal (goal_dim)]`

## Reference

Based on contrastive reinforcement learning principles with independent per-agent training for multi-agent coordination.

