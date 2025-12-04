#!/bin/bash
#SBATCH --job-name=eval_bounded
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=eval_bounded_%j

echo "loading deps"
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

echo "source nvenv"

source $SCRATCH/CSCE-489/CSCE-489-Project/nvenv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/main

export PYTHONPATH="$PYTHONPATH:$SCRATCH/CSCE-489/CSCE-489-Project/infini-gram/pkg/"

echo "running ..."
python eval_bidir_bounded_gram.py \
    --tok_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --fwd_idx_path ../infini-gram/artifacts/train-validate_index \
    --bwd_idx_path ../infini-gram/artifacts/train-validate_reversed_index \
    --dataset_path ../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked/masked_test_random_ds \
    --mask_token_id 2 \
    --max_sequences 300 \
    --output_dir ./test_results_true
