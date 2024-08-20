#!/bin/bash
#SBATCH --job-name=train_moe
#SBATCH -n 4
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=00:40:00
conda activate moe
python /om/user/lshoji/MoE_Interpretability/train_multiple.py n_embd=32 expert_k=2