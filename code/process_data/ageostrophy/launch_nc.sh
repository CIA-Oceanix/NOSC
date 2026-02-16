#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:h100:2 #gpu:a40:3 #gpu:h100:2 #gpu:l40s:4 #gpu:a100:8                       # GPU request
#SBATCH --job-name=uv_ageo_geo                      # Job name
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --output=log/job_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=480G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

srun python compute_geo_ageo.py