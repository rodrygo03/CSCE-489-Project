#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ConvertDataTrain     
#SBATCH --time=00:03:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=3G                  
#SBATCH --output=ConvertDataTrain.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38

echo "Starting conversion..."

python convert_data.py \
  --files ./artifacts/datasets/human_reference_genome/6kbp/0.0.0/1dda7e5c35a9246c397dab1c8343852df92847b32619e1a72261d5cee3b2f919/human_reference_genome-train.arrow \
  --out_dir ./artifacts/datasets/human_reference_genome/6kbp/jsonl/train/ \
  --out_name human_reference_genome-train.jsonl \


## seff 17228407

## Job ID: 17228407
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Cores: 1
## CPU Utilized: 00:00:37
## CPU Efficiency: 82.22% of 00:00:45 core-walltime
## Job Wall-clock time: 00:00:45
## Memory Utilized: 1.81 GB
## Memory Efficiency: 60.39% of 3.00 GB (3.00 GB/node)
