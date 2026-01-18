#!/bin/bash

cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running CRL SMAX 2s3z"

python train_crl_smax_2.py \
  --smax_map_name 2s3z \
  --total_env_steps 250_000_000 \
  --num_epochs 500 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 77 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --exp_name crl_smax_2-sqrt-distance \
  --track 
