#!/bin/bash
# Run IPPO baseline for 3-agent MPE Tag - Seed 1

cd /home/asim_aims_ac_za/gc-marl/baselines/IPPO
export PYTHONPATH="/home/asim_aims_ac_za/gc-marl:$PYTHONPATH"

echo "Starting IPPO 3-agent baseline (Seed 1)..."

python ippo_ff_mpe_facmac.py --config-name ippo_ff_mpe_facmac_paper SEED=1 \
  > ../../logs/ippo_3a_seed1.log 2>&1 &

echo "IPPO 3-agent Seed 1 launched (PID: $!)"
echo "Monitor with: tail -f logs/ippo_3a_seed1.log"

