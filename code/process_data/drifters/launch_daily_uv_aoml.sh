#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:l40s:1 #gpu:a100:1 #gpu:l40s:1 #a100:1 #gpu:rtx8000:3 gpu:a100:8 #gpu:a40:3 #gpu:h100:2 #gpu:rtx8000:3                 # GPU request
#SBATCH --job-name=make_daily_map                      # Job name
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --output=log/make_daily_map.log # Standard output and error log (%j for jobid)
#SBATCH --mem=200G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

echo "make_daily_uv_map_aoml_00m"
srun python make_daily_uv_map_aoml_00m.py