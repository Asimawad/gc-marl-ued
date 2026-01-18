set -euo pipefail

# Neptune token (you can also export it before calling this script)
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNWQwZThhOC0yM2ZiLTRhMWYtODRlNy1mZmI2ZGMyN2ZhNjQifQ=="


# SMAX Environment Configuration
# Available task_names: 2s3z | 3s5z | 5m_vs_6m | 10m_vs_11m | 27m_vs_30m | 3s5z_vs_3s6z | 3s_vs_5z | 6h_vs_8z | smacv2_5_units | smacv2_10_units | smacv2_20_units
ENV="smax"
TASK="smacv2_5_units"  # You can change this to other tasks like 2s3z, 5m_vs_6m, etc.
NUM_ENVS=256  # Increased for 4 TPU cores (64 per core)
timesteps=128000000


# Note: smax_icrl.yaml uses @package _global_, so scenario.task_name is at root level (not env.scenario.task_name)
uv run mava/systems/icrl/anakin/ff_icrl.py -m system.seed=0,1,2,3,4,5 \
    env=$ENV \
    arch.num_envs=$NUM_ENVS \
    system.total_timesteps=$timesteps \
    system.rollout_length=62 \
    system.explore_steps=1000 \
    system.update_batch_size=1 \
    system.episode_length=101 \
    system.buffer_size=5000 \
    system.batch_size=256 \
    system.discrete_actions=False \
    logger.loggers.neptune.enabled=False \
    logger.loggers.neptune.project=Instadeep/ssued \
    logger.loggers.neptune.tag=["ff-icrl-smax","$TASK","seed-0","tpu-4"] \
    logger.loggers.neptune.group_tag=["benchmark-pqn-experiments","$TASK"] \
    system=icrl/ff_icrl \
    arch.num_evaluation=640 \
    arch.num_eval_episodes=100