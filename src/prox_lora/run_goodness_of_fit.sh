#!/bin/bash
#SBATCH --job-name=gof
#SBATCH --partition=common
#SBATCH --qos=kc429229_common
#SBATCH --gres=gpu:1
#SBATCH --exclude=asusgpu3,asusgpu4,asusgpu5,steven
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_eval.log
#SBATCH --error=logs/%j_eval.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kc429229@students.mimuw.edu.pl


# only on some given device   #SBATCH --nodelist=asusgpu1  


# log directory
mkdir -p logs

# print some info
#by hand --exclude=asusgpu4
echo "Running on node: $(hostname)"
nvidia-smi

#export PYTHONUNBUFFERED=1

# run the experiment

uv run src/prox_lora/goodness_of_fit.py
