#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:h100:2                      # GPU request
#SBATCH --job-name=process_pad                       # Job name
#SBATCH --output=./process_%j.log # Standard output and error log (%j for jobid)

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

conda info --env

#folder="/Odyssey/private/t22picar/data/glorys_0m/glorys_multivar_0m_2010-2018.nc"
#folder="/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2019_4th.nc"
#folder="/Odyssey/private/t22picar/data/sst_L4/SST_L4_OSTIA_2019_4th.nc"
#folder="/Odyssey/private/t22picar/data/era5/era5_2019_dailymean_4th.nc"
folder="/Odyssey/private/t22picar/data/glorys_15m/glorys_15.81m_2019-01-01-2020-01-01_4th.nc"

srun python add_pad.py "$folder"
