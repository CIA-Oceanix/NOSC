#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:rtx8000:3                          # GPU request
#SBATCH --job-name=import-sst                       # Job name
#SBATCH --output=./import_sst_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=100G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate base
conda info --env

#srun python import_data_glorys_2020_15m.py > "output.log" 2>&1
srun python import_data_sst_L4.py 
