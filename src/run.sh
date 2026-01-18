#!/usr/bin/env bash
set -euo pipefail

# Neptune token
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNWQwZThhOC0yM2ZiLTRhMWYtODRlNy1mZmI2ZGMyN2ZhNjQifQ=="
source .venv/bin/activate

echo "Running SAC-CRL Baseline"
echo "========================"

ENV="smax"
timesteps=250_000_000 
# 2s3z | 3s5z | 5m_vs_6m | 10m_vs_11m | 27m_vs_30m | 3s5z_vs_3s6z | 3s_vs_5z | 6h_vs_8z | smacv2_5_units | smacv2_10_units | smacv2_20_units
scenario="smacv2_10_units" 

uv run mava/systems/icrl/anakin/sac_crl_brax.py \
    env=$ENV \
    env.scenario.task_name=$scenario \
    env.scenario.map_name=$scenario \
    system.total_timesteps=$timesteps \
    system.actor_lr=3e-4 \
    system.q_lr=3e-4 \
    system.alpha_lr=3e-4 \
    system.max_replay_size=5000 \
    system.min_replay_size=1000 \
    system.rollout_length=62 \
    system.batch_size=256 \
    arch.num_envs=256 \
    logger.loggers.neptune.enabled=true \
    logger.loggers.neptune.project=Instadeep/ssued \
    logger.loggers.neptune.tag=["sac-crl-baseline","$scenario","$ENV"] \
    logger.loggers.neptune.group_tag=["baselines","icrl-sac-baselines","sac-crl-$scenario","$ENV"]

# # ENV="smax"
# timesteps=30_000_000 
# # # 2s3z | 3s5z | 5m_vs_6m | 10m_vs_11m | 27m_vs_30m | 3s5z_vs_3s6z | 3s_vs_5z | 6h_vs_8z | smacv2_5_units | smacv2_10_units | smacv2_20_units
# scenario="smacv2_10_units" 

# python train_icrl_smax.py \
#   seed=1 \
#   env.smax_map_name=$scenario \
#   system.total_env_steps=$timesteps \
#   system.num_epochs=300 \
#   system.num_envs=256 \
#   system.batch_size=256 \
#   logger.loggers.neptune.enabled=True \
#   logger.loggers.console.enabled=True \
#   track=True




# temperature="0.05"
# uv run mava/systems/icrl/anakin/pqn_crl_brax.py \
#     env=$ENV \
#     env.scenario.task_name=$scenario \
#     env.scenario.map_name=$scenario \
#     system.temperature=$temperature \
#     system.total_timesteps=$timesteps \
#     logger.loggers.neptune.enabled=true \
#     logger.checkpointing.save_model=true \
#     logger.checkpointing.save_args.max_to_keep=3 \
#     logger.loggers.neptune.project=Instadeep/ssued \
#     logger.loggers.neptune.tag=["e-greedy"] \
#     logger.loggers.neptune.group_tag=["pqn-optimizing-smax-$scenario","$ENV"] 


# python sac_smax.py \
#   --smax_map_name smacv2_5_units \
#   --total_env_steps 250_000_000 \
#   --num_epochs 500 \
#   --num_envs 256 \
#   --batch_size 256 \
#   --seed 23 \
#   --wandb_project_name ICRL_Reproduction \
#   --wandb_entity asim_awad \
#   --wandb_mode online \
#   --wandb_dir ~/genrl-mara-autocurricula/wandb_logs \
#   --track \


# python train_icrl_smax.py \
#   seed=1 \
#   env.smax_map_name=smacv2_5_units \
#   system.total_env_steps=250_000_000 \
#   system.num_epochs=500 \
#   system.num_envs=256 \
#   system.batch_size=256 \
#   logger.loggers.neptune.enabled=False \
#   logger.loggers.console.enabled=True \
#   track=True
