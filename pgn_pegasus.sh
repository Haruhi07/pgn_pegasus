#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pgn_pegasus
#SBATCH --time=1:0:0
#SBATCH --mem=8192M

cd "${SLURM_SUBMIT_DIR}"

export SAVE_DIR="./cache"

python3 -u ./train.py