#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobMaskData     
#SBATCH --time=01:15:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=15G                  
#SBATCH --output=JobMaskDataOut.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38
python mask_data.py

## to run: sbatch mask_data.slurm


## seff <job id>

## Job ID: 17212844
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Cores: 1
## CPU Utilized: 01:00:12
## CPU Efficiency: 92.90% of 01:04:48 core-walltime
## Job Wall-clock time: 01:04:48
## Memory Utilized: 13.94 GB
## Memory Efficiency: 99.58% of 14.00 GB (14.00 GB/node)
