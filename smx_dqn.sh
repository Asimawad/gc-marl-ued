#!/bin/bash

# Set default values (can override with: MAP=3s5z EPOCHS=100 SEED=2 bash smx_dqn.sh)
MAP=${MAP:-smacv2_5_units}
EPOCHS=${EPOCHS:-500}
SEED=${SEED:-1}

# Use current directory
cd /home/asim/gc-marl-ued
export PYTHONPATH="/home/asim/gc-marl-ued:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running PQN with Learnable Temperature"
echo "  Map: $MAP"
echo "  Epochs: $EPOCHS"
echo "  Seed: $SEED"

python train_pqn_smax_learnable_temp.py \
  --smax_map_name $MAP \
  --num_epochs $EPOCHS \
  --num_envs 512 \
  --initial_temperature 0.2 \
  --target_entropy_ratio 0.3 \
  --min_temperature 0.01 \
  --max_temperature 2.0 \
  --critic_lr 3e-4 \
  --temperature_lr 3e-5 \
  --seed $SEED \
  --exp_name "pqn_learnable_temp_fixed" \
  --track \
  --wandb_mode online \
  --wandb_project_name CRL_SMAX \
  --wandb_entity asim_awad \
  2>&1 | tee logs/pqn_learnable_temp_fixed_seed${SEED}.log
# python train_crl_smax_v2.py \
#   --smax_map_name smacv2_5_units \
#   --total_env_steps 250_000_000 \
#   --num_epochs 500 \
#   --num_envs 256 \
#   --batch_size 256 \
#   --seed 1 \
#   --wandb_project_name ICRL_Reproduction \
#   --wandb_entity asim_awad \
#   --wandb_mode online \
#   --exp_name crl_v2_explicit_action-fixed_temperature_0.0375 \
#   --track 
