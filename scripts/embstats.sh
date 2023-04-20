#!/bin/bash
#SBATCH --job-name=EmbStats
#SBATCH --output=experiments/EmbStats/logs/run.out
#SBATCH --error=experiments/EmbStats/logs/run.err
#SBATCH --time=2-23:00:00
#SBATCH --mem=200GB 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
unset SPLADE_CONFIG_NAME
unset SPLADE_CONFIG_FULLPATH
export SPLADE_CONFIG_NAME="config_splade++_cocondenser_ensembledistil_44G"
python3 scripts/embStats.py
exit