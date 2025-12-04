#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=fine_tune_2gpu_debug
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2        # Test with just 2 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=logs/fine_tune_2gpu_debug.%j

# Load modules for Python + CUDA
module restore deps

# Activate your virtual environment
source $SCRATCH/CSCE-489-Project/venv/bin/activate

cd $SCRATCH/CSCE-489-Project/nucleotide-transformer/fine_tune/code/

# Make sure logs directory exists
mkdir -p logs

echo "==================================="
echo "Multi-GPU Debug Training (2 GPUs)"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: 2"
echo "==================================="

######### Debug run with 2 GPUs, 2 folds, 100 steps #########
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    fine_tune_deepromoter_distributed.py \
    --data-path ../data/nt_promoter_dataset.tsv \
    --output-dir results/promoter_ia3_2gpu_debug \
    --model-name /scratch/user/yojoe56/CSCE-489-Project/nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --n-folds 2 \
    --num-steps 100 \
    --batch-size 8 \
    --learning-rate 3e-3 \
    --test-size 0.2 \
    --seed 42

echo "Debug training complete!"