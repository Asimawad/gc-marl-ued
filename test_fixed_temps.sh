#!/bin/bash
# test_fixed_temps.sh - Quick test of different fixed temperatures

echo "=================================="
echo "Testing Different Fixed Temperatures"
echo "=================================="
echo ""
echo "This will test your ORIGINAL PQN with different temperature values"
echo "Running 50 epochs each for quick comparison"
echo "=================================="

mkdir -p logs

MAP="smacv2_5_units"
EPOCHS=50
SEED=42

# Test different fixed temperatures
for TEMP in 0.05 0.1 0.2 0.5; do
    echo ""
    echo "Testing temperature=$TEMP..."

    python train_pqn_smax.py \
      --smax_map_name $MAP \
      --num_epochs $EPOCHS \
      --num_envs 512 \
      --temperature $TEMP \
      --critic_lr 3e-4 \
      --seed $SEED \
      --exp_name "pqn_fixed_temp_${TEMP}" \
      --track \
      --wandb_mode online \
      --wandb_project_name PQN_Temperature_Search \
      --wandb_entity asim_awad \
      > logs/pqn_temp_${TEMP}_seed${SEED}.log 2>&1 &

    PID=$!
    echo "  Launched with PID $PID"

    # Stagger launches
    sleep 5
done

echo ""
echo "All 4 runs launched in parallel!"
echo ""
echo "Compare results on WandB:"
echo "  https://wandb.ai/asim_awad/PQN_Temperature_Search"
echo ""
echo "Expected results:"
echo "  temp=0.05: ~40-50% (your current result)"
echo "  temp=0.1:  ~50-60% (better exploration)"
echo "  temp=0.2:  ~60-70% (good balance)"
echo "  temp=0.5:  ~55-65% (maybe too much exploration)"
echo ""
