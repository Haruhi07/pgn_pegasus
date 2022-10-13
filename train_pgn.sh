#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=train_pgn
#SBATCH --time=240:0:0
#SBATCH --output=train_pgn.output
#SBATCH --mem=17000M

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u train_pgn.py