#!/bin/bash
#SBATCH --job-name=indexing_train-validate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=indexing_train-validate.%j

module restore deps
source venv/bin/activate

python indexing.py \
  --data_dir $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/jsonl/train-validate/ \
  --save_dir ../../artifacts/train-validate_index/ \
  --tokenizer nucleotide \
  --token_dtype u16 \
  --mem 64 \
  --cpus 8 \
  --shards 1
