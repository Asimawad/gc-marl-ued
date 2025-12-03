#!/bin/bash
# run_smax_2s3z_3seeds.sh - Run ICRL SMAX 2s3z - 3 seeds (sequential)

cd /home/asim/gc-marl-ued
export PYTHONPATH="/home/asim/gc-marl-ued:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================"
echo "Running ICRL SMAX 2s3z - 3 Seeds (Sequential)"
echo "Map: 2s3z (2 Stalkers, 3 Zealots)"
echo "Total steps: 10M per seed (~2 hours each on TPU)"
echo "Expected win rate: >30% (paper reports ~35%)"
echo "================================================"

# Seed 1
echo ""
echo "Starting Seed 1..."
python train_icrl_smax.py \
  --smax_map_name 2s3z \
  --total_env_steps 250000000 \
  --num_epochs 500 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 1 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --track 
  > logs/smax_2s3z_icrl_seed1.log 2>&1 &
