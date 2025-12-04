#!/bin/bash
#SBATCH --job-name=hyperparam_tunining
#SBATCH --time=04:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=hyperparam_tunining_%j

echo "loading deps"
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1


source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/main

export PYTHONPATH="$PYTHONPATH:$SCRATCH/CSCE-489/CSCE-489-Project/infini-gram/pkg/"

echo "running..."
python bidir_infini_gram_v2.py \
    --tok_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --fwd_idx_path ../infini-gram/artifacts/train_index \
    --bwd_idx_path ../infini-gram/artifacts/train_reversed_index \
    --masked_val_path ../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked/masked_val_random_ds \
    --max_val_sequences 25 \
    --max_left 256 \
    --max_right 256 \
    --output hyperparameters.json
echo "done"

## echo "running..."
## python bidir_infini_gram.py \
##     --tok_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
##     --fwd_idx_path ../infini-gram/artifacts/train_index \
##     --bwd_idx_path ../infini-gram/artifacts/train_reversed_index \
##     --masked_val_path ../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked/masked_val_random_ds \
##     --max_val_sequences 25 \
##     --max_left 256 \
##     --max_right 256 \
##     --output hyperparameters.json
## echo "done"
