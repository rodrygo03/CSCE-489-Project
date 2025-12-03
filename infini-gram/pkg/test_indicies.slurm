#!/bin/bash
#SBATCH --job-name=test_indicies
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --output=test_indicies.%j

module restore deps

source venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/infini_gram/"

python test_indicies.py \
  --index_dir ../artifacts/train_index/ \
  --tokenizer_path ../../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref/

echo "***"

python test_indicies.py \
  --index_dir ../artifacts/train_reversed_index/ \
  --tokenizer_path ../../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref/
