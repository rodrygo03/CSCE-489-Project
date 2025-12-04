#!/bin/bash
#SBATCH --job-name=val_hybrid
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 
#SBATCH --output=val_hybrid%j

echo "loading deps"
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

echo "source nvenv"

source $SCRATCH/CSCE-489/CSCE-489-Project/nvenv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/main

export PYTHONPATH="$PYTHONPATH:$SCRATCH/CSCE-489/CSCE-489-Project/infini-gram/pkg/"

echo "running ..."
python val_hybrid.py \
    --tok_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --nt_model_path ../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --fwd_idx_path ../infini-gram/artifacts/train_index \
    --bwd_idx_path ../infini-gram/artifacts/train_reversed_index \
    --max_support_fwd 8 \
    --max_support_bwd 8 \
    --dataset_path ../preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/masked/masked_val_random_ds \
    --max_sequences 20 \
    --output_dir ./h_tuning_results
