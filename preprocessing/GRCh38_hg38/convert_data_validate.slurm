#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ConvertDataValidate     
#SBATCH --time=00:40:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=3G                  
#SBATCH --output=ConvertDataValidate.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38

echo "Starting conversion..."

python convert_data.py \
  --files ./artifacts/datasets/human_reference_genome/6kbp/0.0.0/1dda7e5c35a9246c397dab1c8343852df92847b32619e1a72261d5cee3b2f919/human_reference_genome-validation.arrow \
  --out_dir ./artifacts/datasets/human_reference_genome/6kbp/jsonl/validate/ \
  --out_name human_reference_genome-validate.jsonl \

