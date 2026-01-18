#!/bin/bash
# run_pqn_learnable_temp.sh - Full training run for PQN with learnable temperature

echo "=================================="
echo "PQN with Learnable Temperature - FULL TRAINING"
echo "=================================="
echo ""
echo "Configuration:"
echo "  - Map: smacv2_5_units"
echo "  - Epochs: 500"
echo "  - Num envs: 1024"
echo "  - Initial temperature: 0.5"
echo "  - Target entropy ratio: 0.7"
echo "  - Critic LR: 3e-4"
echo "  - Temperature LR: 1e-4"
echo "  - Seed: 1"
echo ""
echo "This should take several hours..."
echo "=================================="
echo ""

# Create logs directory
mkdir -p logs

# Run training
python train_pqn_smax_learnable_temp.py \
  --smax_map_name smacv2_5_units \
  --total_env_steps 50_000_000 \
  --num_epochs 500 \
  --num_envs 1024 \
  --num_eval_envs 256 \
  --initial_temperature 0.5 \
  --target_entropy_ratio 0.7 \
  --critic_lr 3e-4 \
  --temperature_lr 1e-4 \
  --min_temperature 0.01 \
  --max_temperature 10.0 \
  --batch_size 256 \
  --unroll_length 100 \
  --use_target_network True \
  --target_tau 0.001 \
  --seed 1 \
  --track \
  --wandb_mode online \
  --wandb_project_name CRL_SMAX \
  --wandb_entity asim_awad \
  2>&1 | tee logs/pqn_learnable_temp_full_seed1.log

echo ""
echo "Training complete! Check WandB for results."
echo "Log saved to: logs/pqn_learnable_temp_full_seed1.log"
