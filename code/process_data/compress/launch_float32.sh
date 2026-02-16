#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:rtx8000:3 #gpu:h100:2 #gpu:rtx8000:3           # GPU request
#SBATCH --job-name=float32                        # Job name
#SBATCH --cpus-per-gpu=6    # 12 CPUs for each GPU
#SBATCH --output=log/job_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=130G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

PATH_NC='/Odyssey/private/t22picar/data/glorys_0m/glorys_multivar_0m_2010-2018.nc'

srun python float65_to_float32.py $PATH_NC