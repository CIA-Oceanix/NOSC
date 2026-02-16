#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:h100:1                      # GPU request
#SBATCH --job-name=era5_8th                        # Job name
#SBATCH --output=logs/era5_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=400G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

conda info --env
srun python make_era5_dataset.py
