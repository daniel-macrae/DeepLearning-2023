#!/bin/bash
#SBATCH --job-name=testingSSDLite
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=d.macrae@student.rug.nl
module load Miniconda3
conda create -n SSDLiteEnv python=3.8
source activate SSDLiteEnv
pip install -r requirements.txt
python testHabrok.py --opt "adam" --train-backbone True --epochs 50
python testHabrok.py --train-backbone True --epochs 50
