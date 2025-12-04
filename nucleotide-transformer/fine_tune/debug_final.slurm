#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=fine_tune_torchrun
#SBATCH --time=12:00:00          # Researchers reported ~20 min for 500M model, giving 1h buffer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2        # Request 2 A100 GPUs
#SBATCH --cpus-per-task=32       # All CPUs for the single task
#SBATCH --ntasks=1               # Single task that will spawn 8 processes
#SBATCH --mem=256G               # Total memory
#SBATCH --nodes=1                # Single node with 8 GPUs
#SBATCH --output=logs/fine_tune_torchrun.%j

# Optional email notifications
##SBATCH --mail-type=ALL
##SBATCH --mail-user=yojoe56@tamu.edu

# Load modules for Python + CUDA
module restore deps

# Activate your virtual environment
source $SCRATCH/CSCE-489-Project/venv/bin/activate

cd $SCRATCH/CSCE-489-Project/nucleotide-transformer/fine_tune/code/

# Make sure logs directory exists
mkdir -p logs

echo "==================================="
echo "Multi-GPU Training Configuration (torchrun)"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "==================================="

######### Fine-tune Nucleotide Transformer with IA3 on 8 A100 GPUs #########
# Using torchrun for launching distributed training
# torchrun automatically sets up RANK, WORLD_SIZE, LOCAL_RANK, etc.

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    fine_tune_deepromoter_distributed_full.py \
    --data-path ../data/nt_promoter_dataset.tsv \
    --output-dir results/promoter_ia3_8gpu_torchrun \
    --model-name /scratch/user/yojoe56/CSCE-489-Project/nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --n-folds 30 \
    --num-steps 10000 \
    --batch-size 8 \
    --learning-rate 3e-3 \
    --test-size 0.2 \
    --seed 42

echo "Training complete!"