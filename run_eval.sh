#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pgn_pegasus
#SBATCH --time=24:0:0
#SBATCH --mem=8192M

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u ./eval.py --dataset ./dataset_cache