#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pgn_pegasus
#SBATCH --time=1:0:0
#SBATCH --mem=8192M
#SBATCH --output=gen_dataset.output

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u ./generate_dataset.py