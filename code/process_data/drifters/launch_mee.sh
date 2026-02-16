#!/bin/bash
#SBATCH --partition=Mee                         # Partition name                      # GPU request
#SBATCH --job-name=lauch_mee                        # Job name
#SBATCH --output=log/job_%j.log # Standard output and error log (%j for jobid)

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

srun python count_drifter.py