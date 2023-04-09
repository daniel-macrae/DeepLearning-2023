#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
module load Miniconda3
conda create -n SSDLiteEnv python=3.8
source activate SSDLiteEnv
pip install -r requirements.txt
python testHabrok.py