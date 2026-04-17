#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=common
#SBATCH --qos=kc429229_common
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_eval.log
#SBATCH --error=logs/%j_eval.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kc429229@students.mimuw.edu.pl


# log directory
mkdir -p logs

# print some info
echo "Running on node: $(hostname)"
nvidia-smi

export PYTHONUNBUFFERED=1

# run the experiment
uv run src/prox_lora/robustness_eval.py