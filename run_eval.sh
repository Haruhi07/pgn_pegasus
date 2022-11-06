#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_short
#SBATCH --job-name=eval_0
#SBATCH --time=6:0:0
#SBATCH --mem=16384M

cd "${SLURM_SUBMIT_DIR}"

source venv/bin/activate
python3 -u ./eval.py --dataset ./dataset_cache --checkpoint google/pegasus-cnn_dailymail --epoch cnndm
