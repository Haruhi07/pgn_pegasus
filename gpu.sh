#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=gpu_stat
#SBATCH --time=0:10:0
#SBATCH --mem=1000M
#SBATCH --output=gpu.output

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u ./gpu.py