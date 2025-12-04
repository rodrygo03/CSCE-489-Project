#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ConvertDataMValidate     
#SBATCH --time=00:40:00              
#SBATCH --ntasks=1                   
#SBATCH --mem=3G                  
#SBATCH --output=ConvertDataMValidate.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38

echo "Starting conversion..."

python convert_data.py \
  --files ./artifacts/datasets/human_reference_genome/6kbp/masked/masked_val_random_ds/data-00000-of-00001.arrow \
  --out_dir ./artifacts/datasets/human_reference_genome/6kbp/jsonl/validate/ \
  --out_name human_reference_genome-mvalidate.jsonl \

