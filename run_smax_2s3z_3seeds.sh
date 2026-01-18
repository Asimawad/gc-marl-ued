#!/bin/bash
# run_smax_3s3z_3seeds.sh - Run ICRL SMAX 3s3z - 3 seeds (sequential)

cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running ICRL SMAX 3s3z"

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
#   sf logs/smax_3s3z_icrl_seed1.log"

# # Wait for seed 1 to finish
# echo "Waiting for Seed 1 to complete..."
# wait $SEED1_PID
# echo "Seed 1 completed!"

# # Seed 2
# echo ""
# echo "Starting Seed 2..."
# python train_icrl_smax.py \
#   --smax_map_name 2s3z \
#   --total_env_steps 10000000 \
#   --num_epochs 100 \
#   --num_envs 256 \
#   --batch_size 256 \
#   --seed 2 \
#   --wandb_project_name ICRL_Reproduction \
#   --wandb_entity asim_awad \
#   --wandb_mode online \
#   --track \
#   > logs/smax_2s3z_icrl_seed2.log 2>&1 &

# SEED2_PID=$!
# echo "Seed 2 launched (PID: $SEED2_PID)"
# echo "Monitor: tail -f logs/smax_2s3z_icrl_seed2.log"

# # Wait for seed 2 to finish
# echo "Waiting for Seed 2 to complete..."
# wait $SEED2_PID
# echo "Seed 2 completed!"

# # Seed 3
# echo ""
# echo "Starting Seed 3..."
# python train_icrl_smax.py \
#   --smax_map_name 2s3z \
#   --total_env_steps 10000000 \
#   --num_epochs 100 \
#   --num_envs 256 \
#   --batch_size 256 \
#   --seed 3 \
#   --wandb_project_name ICRL_Reproduction \
#   --wandb_entity asim_awad \
#   --wandb_mode online \
#   --track \
#   > logs/smax_2s3z_icrl_seed3.log 2>&1 &

# SEED3_PID=$!
# echo "Seed 3 launched (PID: $SEED3_PID)"
# echo "Monitor: tail -f logs/smax_2s3z_icrl_seed3.log"

# # Wait for seed 3 to finish
# echo "Waiting for Seed 3 to complete..."
# wait $SEED3_PID
# echo "Seed 3 completed!"

# echo ""
# echo "================================================"
# echo "All 3 seeds completed!"
# echo "Total runtime: ~6 hours"
# echo "Logs saved in logs/ directory"
# echo "Check WandB for results comparison"
# echo "================================================"
