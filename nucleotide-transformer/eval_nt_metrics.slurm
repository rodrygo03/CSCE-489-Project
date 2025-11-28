#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=eval_nt_metrics
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=2G   
#SBATCH --output=eval_nt_metrics.%j

module restore deps
source $SCRATCH/CSCE-489/CSCE-489-Project/venv/bin/activate
cd $SCRATCH/CSCE-489/CSCE-489-Project/nucleotide-transformer
python eval_nt_metrics.py


## seff <job id>

## Job ID: 17214148
## Cluster: grace
## User/Group: rodrigoorozco03/rodrigoorozco03
## State: COMPLETED (exit code 0)
## Nodes: 1
## Cores per node: 4
## CPU Utilized: 00:25:34
## CPU Efficiency: 19.83% of 02:08:56 core-walltime
## Job Wall-clock time: 00:32:14
## Memory Utilized: 1010.64 MB
## Memory Efficiency: 4.11% of 24.00 GB (24.00 GB/node)
