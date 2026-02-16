#!/bin/bash
#SBATCH --partition=Odyssey                         # Partition name
#SBATCH --gres=gpu:l40s:4                      # GPU request
#SBATCH --job-name=proc_era5                        # Job name
#SBATCH --output=./process_era5_%j.log # Standard output and error log (%j for jobid)
#SBATCH --mem=50G

export HOME=/Odyssey/private/t22picar/
source "/Odyssey/private/t22picar/miniforge3/etc/profile.d/conda.sh"
conda activate 4dvarnet-daniel

conda info --env

for year in {2010..2018}

do 
    echo $year
    srun python era5_8th_year.py "$year"
done


