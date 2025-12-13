#!/bin/bash
# run_ippo_smax_2s3z.sh - Run IPPO baseline for SMAX 2s3z

cd /home/asim/gc-marl-ued
export PYTHONPATH="/home/asim/gc-marl-ued:$PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================"
echo "Running IPPO Baseline for SMAX 2s3z"
echo "Map: 2s3z (2 Stalkers, 3 Zealots)"
echo "Total steps: 10M"
echo "Expected win rate: ~0% (baseline comparison)"
echo "================================================"

# Run IPPO baseline
echo ""
echo "Starting IPPO baseline (Seed 1)..."
cd baselines/IPPO
python ippo_no_rnn_smax.py \
  --config-name ippo_no_rnn_smax \
  SEED=1 \
  MAP_NAME=2s3z \
  TOTAL_TIMESTEPS=50000000 \
  ENTITY=asim_awad \
  PROJECT=ICRL_Reproduction \
  WANDB_MODE=online \
  > ../../logs/ippo_smax_2s3z_seed1.log 2>&1 &

IPPO_PID=$!
echo "IPPO baseline launched (PID: $IPPO_PID)"
echo "Monitor: tail -f logs/ippo_smax_2s3z_seed1.log"
echo ""
echo "To run additional seeds, use:"
echo "  python ippo_no_rnn_smax.py --config-name ippo_no_rnn_smax SEED=2 ..."
echo "  python ippo_no_rnn_smax.py --config-name ippo_no_rnn_smax SEED=3 ..."

