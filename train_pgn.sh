#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=train_pgn
#SBATCH --time=0:10:0
#SBATCH --mem=8192M

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u train_pgn.py