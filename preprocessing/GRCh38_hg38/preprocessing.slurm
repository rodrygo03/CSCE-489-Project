#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobPreprocessing     
#SBATCH --time=00:30:00              
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=2560M                  #Request 2560MB (2.5GB) per task
#SBATCH --output=JobPreprocessingOut.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate

cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38

python mask_data.py


## to run: sbatch mast_data.py
