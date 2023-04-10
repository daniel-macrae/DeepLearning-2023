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
python testHabrok.py --opt "adam"
python testHabrok.py --opt "adam" --lr 0.05
python testHabrok.py
python testHabrok.py --momentum 0.5
python testHabrok.py --momentum 0.1
python testHabrok.py --lr-step-size 20
python testHabrok.py --lr-step-size 5
python testHabrok.py --lr-step-size 5 lr-gamma 0.05
python testHabrok.py lr-gamma 0.3
python testHabrok.py --lr 0.05
python testHabrok.py --lr 0.001