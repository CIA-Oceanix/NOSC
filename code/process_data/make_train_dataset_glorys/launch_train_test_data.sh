#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:l40s:1 #gpu:h100:1 #gpu:a40:3 #gpu:l40s:1 #gpu:h100:1 #gpu:a100:8 #gpu:rtx8000:3 #gpu:h100:2 #gpu:a100:8 #gpu:rtx8000:3 #gpu:a100:8 #gpu:h100:2 #gpu:l40s:4 #gpu:a100:8  #gpu:h100:2 #gpu:rtx8000:3 #gpu:a40:3 #gpu:rtx8000:3 #gpu:a100:8 #gpu:h100:2 #gpu:l40s:4 #gpu:a100:8  #gpu:h100:2 #gpu:a100:8                       # GPU request
#SBATCH --job-name=make_train_test_data                  # Job name
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --output=log/job_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=300G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

srun python make_train_dataset_glorys.py
srun python make_test_dataset_glorys.py