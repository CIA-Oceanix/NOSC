#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:a100:1                 # GPU request
#SBATCH --job-name=make_daily_map                      # Job name
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --output=log/make_daily_map.log # Standard output and error log (%j for jobid)
#SBATCH --mem=200G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

srun python make_daily_uv_map_cmems_00m.py