#!/bin/bash
#SBATCH --job-name=gams
#SBATCH --output=logs/gams-%j.out
#SBATCH --error=logs/gams-%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00

singularity exec --nv ../containers/singularity_python3.10_classla.sif bash -c "
python finetuning_preprocessing_classla.py
"
