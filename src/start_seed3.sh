#!/bin/bash
cd /home/asim_aims_ac_za/gc-marl
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

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

echo "Seed 3 launched (PID: $!)"
echo "Monitor with: tail -f logs/mpe_6a_icrl_seed3.log"

