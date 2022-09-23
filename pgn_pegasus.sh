#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pgn_pegasus
#SBATCH --time=0:20:0
#SBATCH --mem=8192M

cd /user/work/hs20307
python3 -u ./generate_dataset.py