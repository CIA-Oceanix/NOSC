#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:rtx8000:1 #gpu:h100:1 #gpu:rtx8000:3                          # GPU request
#SBATCH --job-name=import-gc                       # Job name
#SBATCH --output=log/gc.log # Standard output and error log (%j for jobid)
#SBATCH --mem=30G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate base
conda info --env

#srun python import_data_glorys_2020_15m.py > "output.log" 2>&1
srun python import_data_copernicus.py 
