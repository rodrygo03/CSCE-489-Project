#!/bin/bash
#SBATCH --job-name=bounded_hyperparam_tunining
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=bounded_hyperparam_tunining_%j

echo "loading deps"
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/main

export PYTHONPATH="$PYTHONPATH:$SCRATCH/CSCE-489/CSCE-489-Project/infini-gram/pkg/"

python val_bidir_bounded_gram.py \
    --tok_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --fwd_idx_path ../infini-gram/artifacts/train_index \
    --bwd_idx_path ../infini-gram/artifacts/train_reversed_index \
    --dataset_path ../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked/masked_val_random_ds \
    --mask_token_id 2 \
    --max_sequences 1000 \
    --output_dir ./tuning_results_2
