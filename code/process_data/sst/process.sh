#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:a100:8                      # GPU request
#SBATCH --job-name=proc_sst                       # Job name
#SBATCH --output=./process_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=300G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

conda info --env
srun python sst_8th.py
