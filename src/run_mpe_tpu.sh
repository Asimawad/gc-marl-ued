#!/bin/bash
# MPE Tag training with TPU v4 support
set -e

cd /home/asim_aims_ac_za/gc-marl
source .venv/bin/activate

echo "Starting MPE Tag training on TPU v4..."
echo "TPU devices: $(python -c 'import jax; print(len(jax.devices()))')"

python train_icrl.py \
  --env_id mpe_tag_facmac \
  --total_env_steps 5000000 \
  --num_epochs 100 \
  --num_envs 256 \
  --num_eval_envs 64 \
  --batch_size 256 \
  --seed 1 \
  --wandb_project_name ICRL_TPU_Validation \
  --wandb_mode offline \
  --track

