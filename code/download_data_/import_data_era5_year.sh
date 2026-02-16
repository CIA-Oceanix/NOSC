#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:rtx8000:1                         # GPU request
#SBATCH --job-name=import-data                       # Job name
#SBATCH --output=./era5_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=30G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate base
conda info --env

for year in {2014..2014}

do 
    echo $year
    srun python import_data_era5_neu_1y.py "$year"
done