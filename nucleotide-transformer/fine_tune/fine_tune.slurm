#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=fine_tune
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --output=logs/fine_tune.%j

# Optional email notifications
##SBATCH --mail-type=ALL
##SBATCH --mail-user=yojoe56@tamu.edu

# Load modules for Python + CUDA (example from Grace docs)
module restore deps
# ^ Adjust if your group uses a specific module stack for DL :contentReference[oaicite:4]{index=4}

# Activate your virtual environment
source $SCRATCH/CSCE-489-Project/venv/bin/activate    # <-- change to your venv path

cd $SCRATCH/CSCE-489-Project/nucleotide-transformer/fine_tune/code/ # path to fine tunning

# Make sure logs directory exists
# mkdir -p logs



######### (2) Fine-tune Nucleotide Transformer with IA3 #########
python fine_tune_deepromoter.py \
    --data-path ../data/nt_promoter_dataset.tsv \
    --output-dir results/promoter_ia3_debug \
    --model-name /scratch/user/yojoe56/CSCE-489-Project/nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
    --n-folds 2 \
    --num-steps 2 \
    --batch-size 2 \
    --learning-rate 3e-3 \
    --test-size 0.2 \
    --seed 42
# python fine_tune_deepromoter.py \
#     --data-path ../data/nt_promoter_dataset.tsv \
#     --output-dir results/promoter_ia3 \
#     --model-name /scratch/user/yojoe56/CSCE-489-Project/nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref \
#     --n-folds 10 \
#     --num-steps 10000 \
#     --batch-size 8 \
#     --learning-rate 3e-3 \
#     --test-size 0.2 \
#     --seed 42
