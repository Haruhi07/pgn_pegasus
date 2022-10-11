#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=run_validation
#SBATCH --time=1:0:0
#SBATCH --mem=8192M
#SBATCH --output=validation.output

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u ./validation.py