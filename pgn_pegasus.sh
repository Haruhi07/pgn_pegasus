#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pgn_pegasus
#SBATCH --time=1:0:0
#SBATCH --mem=8192M

cd "${SLURM_SUBMIT_DIR}"
python3 -u ./generate_dataset.py