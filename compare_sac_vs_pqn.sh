#!/bin/bash
# compare_sac_vs_pqn.sh - Compare SAC baseline vs PQN with learnable temperature

echo "=================================="
echo "SAC vs PQN Comparison"
echo "=================================="
echo ""
echo "This will run:"
echo "  1. SAC baseline (train_icrl_smax.py)"
echo "  2. PQN with learnable temperature (train_pqn_smax_learnable_temp.py)"
echo "  3. Original PQN with fixed temp (train_pqn_smax.py)"
echo ""
echo "All with same seed for fair comparison"
echo "=================================="

mkdir -p logs

SEED=42
MAP="smacv2_5_units"
EPOCHS=100  # Shorter for quick comparison

# Run SAC baseline
echo ""
echo "[1/3] Running SAC baseline..."
python train_icrl_smax.py \
  --smax_map_name $MAP \
  --num_epochs $EPOCHS \
  --num_envs 256 \
  --batch_size 256 \
  --seed $SEED \
  --exp_name "sac_baseline_comparison" \
  --track \
  --wandb_mode online \
  --wandb_project_name SAC_vs_PQN_Comparison \
  --wandb_entity asim_awad \
  > logs/comparison_sac_seed${SEED}.log 2>&1 &

SAC_PID=$!
echo "SAC launched (PID: $SAC_PID)"

# Wait a bit to stagger launches
sleep 10

# Run PQN with learnable temperature
echo ""
echo "[2/3] Running PQN with learnable temperature..."
python train_pqn_smax_learnable_temp.py \
  --smax_map_name $MAP \
  --num_epochs $EPOCHS \
  --num_envs 512 \
  --initial_temperature 0.5 \
  --target_entropy_ratio 0.7 \
  --critic_lr 3e-4 \
  --seed $SEED \
  --exp_name "pqn_learnable_temp_comparison" \
  --track \
  --wandb_mode online \
  --wandb_project_name SAC_vs_PQN_Comparison \
  --wandb_entity asim_awad \
  > logs/comparison_pqn_learnable_seed${SEED}.log 2>&1 &

PQN_LEARN_PID=$!
echo "PQN (learnable temp) launched (PID: $PQN_LEARN_PID)"

# Wait a bit
sleep 10

# Run original PQN with fixed temperature
echo ""
echo "[3/3] Running PQN with fixed temperature (baseline)..."
python train_pqn_smax.py \
  --smax_map_name $MAP \
  --num_epochs $EPOCHS \
  --num_envs 512 \
  --temperature 0.05 \
  --critic_lr 1e-4 \
  --seed $SEED \
  --exp_name "pqn_fixed_temp_comparison" \
  --track \
  --wandb_mode online \
  --wandb_project_name SAC_vs_PQN_Comparison \
  --wandb_entity asim_awad \
  > logs/comparison_pqn_fixed_seed${SEED}.log 2>&1 &

PQN_FIXED_PID=$!
echo "PQN (fixed temp) launched (PID: $PQN_FIXED_PID)"

echo ""
echo "=================================="
echo "All three runs launched in parallel!"
echo "=================================="
echo ""
echo "Monitor progress:"
echo "  SAC:              tail -f logs/comparison_sac_seed${SEED}.log"
echo "  PQN (learnable):  tail -f logs/comparison_pqn_learnable_seed${SEED}.log"
echo "  PQN (fixed):      tail -f logs/comparison_pqn_fixed_seed${SEED}.log"
echo ""
echo "View on WandB:"
echo "  https://wandb.ai/asim_awad/SAC_vs_PQN_Comparison"
echo ""
echo "Waiting for all runs to complete..."
echo ""

# Wait for all to finish
wait $SAC_PID
echo "✓ SAC completed"

wait $PQN_LEARN_PID
echo "✓ PQN (learnable temp) completed"

wait $PQN_FIXED_PID
echo "✓ PQN (fixed temp) completed"

echo ""
echo "=================================="
echo "All runs completed!"
echo "=================================="
echo ""
echo "Compare results on WandB to see:"
echo "  - Did learnable temperature help?"
echo "  - How close is PQN to SAC baseline?"
echo ""
