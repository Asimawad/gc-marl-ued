#!/bin/bash
# run_smax_smacv2_5_units_3seeds.sh - Run ICRL SMAX smacv2_5_units - 3 seeds (sequential)

cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running ICRL SMAX smacv2_5_units"

python train_icrl_smax.py \
  --smax_map_name smacv2_5_units \
  --total_env_steps 250_000_000 \
  --num_epochs 500 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 1 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --track \
  > logs/smax_smacv2_5_units_icrl_seed1.log 2>&1 &

SEED1_PID=$!
echo "Seed 1 launched (PID: $SEED1_PID)"
echo "Monitor: tail -f logs/smax_smacv2_5_units_icrl_seed1.log"