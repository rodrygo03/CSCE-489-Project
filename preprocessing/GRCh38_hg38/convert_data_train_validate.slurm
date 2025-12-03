#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ConvertDataTrainVali     
#SBATCH --time=00:05:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=5G                  
#SBATCH --output=ConvertDataTrainVali.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38

echo "Starting conversion..."

python convert_data.py \
  --files ./artifacts/datasets/human_reference_genome/6kbp/0.0.0/1dda7e5c35a9246c397dab1c8343852df92847b32619e1a72261d5cee3b2f919/human_reference_genome-train.arrow \
  ./artifacts/datasets/human_reference_genome/6kbp/0.0.0/1dda7e5c35a9246c397dab1c8343852df92847b32619e1a72261d5cee3b2f919/human_reference_genome-validation.arrow \
  --out_dir ./artifacts/datasets/human_reference_genome/6kbp/jsonl/train-validate \
  --out_name human_reference_genome-train-validation.jsonl \


## seff 17228408

## Job ID: 17228408
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Cores: 1
## CPU Utilized: 00:00:38
## CPU Efficiency: 82.61% of 00:00:46 core-walltime
## Job Wall-clock time: 00:00:46
## Memory Utilized: 1.80 GB
## Memory Efficiency: 35.98% of 5.00 GB (5.00 GB/node)
