#!/bin/bash
#SBATCH --job-name=indexing_train
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=indexing_train.%j

module restore deps
source venv/bin/activate

python indexing.py \
  --data_dir $SCRATCH/CSCE-489/CSCE-489-Project/preprocessing/GRCh38_hg38/artifacts/datasets/human_reference_genome/6kbp/jsonl/train/ \
  --save_dir ../../artifacts/train_index/ \
  --tokenizer nucleotide \
  --token_dtype u16 \
  --mem 64 \
  --cpus 8 \
  --shards 1


## Job ID: 17228439
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Nodes: 1
## Cores per node: 8
## CPU Utilized: 01:01:27
## CPU Efficiency: 38.03% of 02:41:36 core-walltime
## Job Wall-clock time: 00:20:12
## Memory Utilized: 6.41 GB
## Memory Efficiency: 10.02% of 64.00 GB (64.00 GB/node)
