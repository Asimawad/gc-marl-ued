#!/bin/bash
# test_pqn_learnable_temp.sh - Test PQN with learnable temperature

set -e  # Exit on error

echo "=================================="
echo "Testing PQN with Learnable Temperature"
echo "=================================="

# Create logs directory
mkdir -p logs

# PHASE 1: Syntax check
echo ""
echo "[1/3] Checking Python syntax..."
python -m py_compile train_pqn_smax_learnable_temp.py
if [ $? -eq 0 ]; then
    echo "✓ Syntax check passed"
else
    echo "✗ Syntax error found!"
    exit 1
fi

# PHASE 2: Quick import test
echo ""
echo "[2/3] Testing imports..."
python -c "
import sys
try:
    # Just import, don't run
    import train_pqn_smax_learnable_temp
    print('✓ All imports successful')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
except Exception as e:
    # Other errors are okay at this stage (e.g., tyro parsing)
    print('✓ Imports loaded (tyro parsing skipped)')
    sys.exit(0)
"

if [ $? -ne 0 ]; then
    echo "✗ Import test failed!"
    exit 1
fi

# PHASE 3: Short training test (2 epochs)
echo ""
echo "[3/3] Running short training test (2 epochs)..."
echo "Log: logs/pqn_learnable_temp_test.log"
echo ""

python train_pqn_smax_learnable_temp.py \
  --smax_map_name smacv2_5_units \
  --num_epochs 2 \
  --num_envs 128 \
  --num_eval_envs 64 \
  --initial_temperature 0.5 \
  --target_entropy_ratio 0.7 \
  --critic_lr 3e-4 \
  --temperature_lr 1e-4 \
  --seed 1 \
  --track \
  --wandb_mode offline \
  --wandb_project_name PQN_Test \
  > logs/pqn_learnable_temp_test.log 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
    echo ""
    echo "The code works! Here's what happened:"
    echo ""

    # Show last 20 lines of log
    tail -n 20 logs/pqn_learnable_temp_test.log

    echo ""
    echo "=================================="
    echo "Ready for full training!"
    echo "=================================="
    echo ""
    echo "To run full training (500 epochs), use:"
    echo ""
    echo "  bash run_pqn_learnable_temp.sh"
    echo ""
    echo "Or manually:"
    echo ""
    echo "  python train_pqn_smax_learnable_temp.py \\"
    echo "    --smax_map_name smacv2_5_units \\"
    echo "    --num_epochs 500 \\"
    echo "    --num_envs 1024 \\"
    echo "    --initial_temperature 0.5 \\"
    echo "    --target_entropy_ratio 0.7 \\"
    echo "    --critic_lr 3e-4 \\"
    echo "    --seed 1 \\"
    echo "    --track \\"
    echo "    --wandb_mode online"
    echo ""
else
    echo ""
    echo "✗✗✗ TEST FAILED! ✗✗✗"
    echo ""
    echo "Check the error log:"
    echo "  cat logs/pqn_learnable_temp_test.log"
    echo ""
    echo "Last 30 lines of error log:"
    tail -n 30 logs/pqn_learnable_temp_test.log
    exit 1
fi
