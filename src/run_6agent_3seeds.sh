#!/bin/bash
# Run ICRL 6-agent (fixed config: 6 predators, 1 prey, 2 landmarks) - 3 seeds

cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

echo "================================================"
echo "Running ICRL 6-Agent (Fixed Config) - 3 Seeds"
echo "Config: 6 predators, 1 prey, 2 landmarks"
echo "Expected returns: ~12,000-15,000 (matching paper)"
echo "================================================"

# Seed 1
echo ""
echo "Starting Seed 1..."
python train_icrl.py \
  --env_id mpe_tag_facmac_6a \
  --total_env_steps 20000000 \
  --num_epochs 200 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 1 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --track \
  > logs/mpe_6a_icrl_seed1.log 2>&1 &

SEED1_PID=$!
echo "Seed 1 launched (PID: $SEED1_PID)"

# Wait for seed 1 to finish (or run in parallel by commenting this out)
echo "Waiting for Seed 1 to complete..."
wait $SEED1_PID
echo "Seed 1 completed!"

# Seed 2
echo ""
echo "Starting Seed 2..."
python train_icrl.py \
  --env_id mpe_tag_facmac_6a \
  --total_env_steps 20000000 \
  --num_epochs 200 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 2 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --track \
  > logs/mpe_6a_icrl_seed2.log 2>&1 &

SEED2_PID=$!
echo "Seed 2 launched (PID: $SEED2_PID)"

# Wait for seed 2 to finish
echo "Waiting for Seed 2 to complete..."
wait $SEED2_PID
echo "Seed 2 completed!"

# Seed 3
echo ""
echo "Starting Seed 3..."
python train_icrl.py \
  --env_id mpe_tag_facmac_6a \
  --total_env_steps 20000000 \
  --num_epochs 200 \
  --num_envs 256 \
  --batch_size 256 \
  --seed 3 \
  --wandb_project_name ICRL_Reproduction \
  --wandb_entity asim_awad \
  --wandb_mode online \
  --track \
  > logs/mpe_6a_icrl_seed3.log 2>&1 &

SEED3_PID=$!
echo "Seed 3 launched (PID: $SEED3_PID)"

# Wait for seed 3 to finish
echo "Waiting for Seed 3 to complete..."
wait $SEED3_PID
echo "Seed 3 completed!"

echo ""
echo "================================================"
echo "All 3 seeds completed!"
echo "Logs saved in logs/ directory"
echo "================================================"

