#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=common
#SBATCH --qos=kc429229_common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval.log
#SBATCH --error=logs/%j_eval.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kc429229@students.mimuw.edu.pl
#SBATCH --nodelist=asusgpu1  


# log directory
mkdir -p logs

# print some info
#by hand --exclude=asusgpu4
echo "Running on node: $(hostname)"
nvidia-smi

#export PYTHONUNBUFFERED=1

# run the experiment
#uv run src/prox_lora/robustness_eval_noise.py
uv run goodness_of_fit.py