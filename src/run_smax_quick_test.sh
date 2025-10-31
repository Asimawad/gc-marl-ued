#!/bin/bash
# Quick validation test for SMAX 2s3z - optimized for CPU
# Reduced parameters for faster execution while still testing the algorithm

cd /home/asim_aims_ac_za/gc-marl

python train_icrl_smax.py \
  --smax_map_name 2s3z \
  --total_env_steps 5000000 \
  --num_epochs 50 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 128 \
  --seed 1 \
  --wandb_project_name ICRL_CPU_Validation \
  --wandb_mode offline \
  --track True

