#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobMaskData     
#SBATCH --time=02:00:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=14G                  
#SBATCH --output=JobMaskDataOut.%j      #Send stdout/err to "Example4Out.[jobID]"

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38
python mask_data.py

## to run: sbatch mask_data.slurm


## seff 17115348

## Job ID: 17115348
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Cores: 1
## CPU Utilized: 01:00:07
## CPU Efficiency: 65.07% of 01:32:23 core-walltime
## Job Wall-clock time: 01:32:23
## Memory Utilized: 11.79 GB
## Memory Efficiency: 73.71% of 16.00 GB (16.00 GB/node)
