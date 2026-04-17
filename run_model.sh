#!/bin/bash
#SBATCH --job-name=CNN_Adam
#SBATCH --partition=common
#SBATCH --qos=kc429229_common
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_train.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kc429229@students.mimuw.edu.pl


# log directory TODO: should we have different logging for train and for eval and for robustness eval?
mkdir -p logs

# print some info
echo "Running on node: $(hostname)"
nvidia-smi

# run the experiment
uv run src/prox_lora/train.py