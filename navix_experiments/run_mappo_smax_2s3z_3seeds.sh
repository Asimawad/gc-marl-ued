#!/bin/bash
# run_mappo_smax_2s3z_3seeds.sh - Run MAPPO SMAX 2s3z baseline - 3 seeds (sequential)

cd /home/asim_aims_ac_za/gc-marl/baselines/MAPPO
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p ../../logs

echo "========================================================"
echo "Running MAPPO SMAX 2s3z Baseline - 3 Seeds (Sequential)"
echo "Map: 2s3z (2 Stalkers, 3 Zealots)"
echo "Total steps: 10M per seed (~2-3 hours each on TPU)"
echo "Expected: Paper shows MAPPO reaches ~45% win rate"
echo "========================================================"

# Seed 1
echo ""
echo "Starting MAPPO Seed 1..."
conda run -n gcmarl python mappo_rnn_smax.py \
  MAP_NAME=2s3z \
  TOTAL_TIMESTEPS=10000000 \
  NUM_ENVS=256 \
  NUM_STEPS=128 \
  SEED=1 \
  ENTITY=asim_awad \
  PROJECT=ICRL_Reproduction \
  WANDB_MODE=online \
  > ../../logs/mappo_smax_2s3z_seed1.log 2>&1 &

SEED1_PID=$!
echo "Seed 1 launched (PID: $SEED1_PID)"
echo "Monitor: tail -f ../../logs/mappo_smax_2s3z_seed1.log"

# Wait for seed 1 to finish
echo "Waiting for Seed 1 to complete..."
wait $SEED1_PID
echo "Seed 1 completed!"

# Seed 2
echo ""
echo "Starting MAPPO Seed 2..."
conda run -n gcmarl python mappo_rnn_smax.py \
  MAP_NAME=2s3z \
  TOTAL_TIMESTEPS=10000000 \
  NUM_ENVS=256 \
  NUM_STEPS=128 \
  SEED=2 \
  ENTITY=asim_awad \
  PROJECT=ICRL_Reproduction \
  WANDB_MODE=online \
  > ../../logs/mappo_smax_2s3z_seed2.log 2>&1 &

SEED2_PID=$!
echo "Seed 2 launched (PID: $SEED2_PID)"
echo "Monitor: tail -f ../../logs/mappo_smax_2s3z_seed2.log"

# Wait for seed 2 to finish
echo "Waiting for Seed 2 to complete..."
wait $SEED2_PID
echo "Seed 2 completed!"

# Seed 3
echo ""
echo "Starting MAPPO Seed 3..."
conda run -n gcmarl python mappo_rnn_smax.py \
  MAP_NAME=2s3z \
  TOTAL_TIMESTEPS=10000000 \
  NUM_ENVS=256 \
  NUM_STEPS=128 \
  SEED=3 \
  ENTITY=asim_awad \
  PROJECT=ICRL_Reproduction \
  WANDB_MODE=online \
  > ../../logs/mappo_smax_2s3z_seed3.log 2>&1 &

SEED3_PID=$!
echo "Seed 3 launched (PID: $SEED3_PID)"
echo "Monitor: tail -f ../../logs/mappo_smax_2s3z_seed3.log"

# Wait for seed 3 to finish
echo "Waiting for Seed 3 to complete..."
wait $SEED3_PID
echo "Seed 3 completed!"

echo ""
echo "========================================================"
echo "All 3 MAPPO seeds completed!"
echo "Total runtime: ~6-9 hours"
echo "Logs saved in logs/ directory"
echo "Check WandB for MAPPO vs ICRL comparison"
echo "Expected: MAPPO ~45% win rate, ICRL should exceed this"
echo "========================================================"

