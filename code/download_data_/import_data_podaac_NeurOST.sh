#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:rtx8000:1                          # GPU request
#SBATCH --job-name=import_neurost                       # Job name
#SBATCH --output=./log/import_neurost.log # Standard output and error log (%j for jobid)
#SBATCH --mem=8G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda base

conda info --env
srun import_data_neurost_sst_ssh.sh
#srun podaac-data-downloader -c NEUROST_SSH-SST_L4_V2024.0 -d /Odyssey/public/NeurOST/2010-2020 --start-date 2010-01-01T00:00:00Z --end-date 2020-01-01T00:00:00Z -e ""