#!/bin/bash

cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running PQN SMAX smacv2_5_units"

python train_pqn_smax.py \
  --smax_map_name smacv2_5_units \
  --total_env_steps 250_000_000 \
  --num_epochs 500 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 77 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --exp_name pqn_smax-tau-0 \
  --target_tau 0.005 \
  --track 
