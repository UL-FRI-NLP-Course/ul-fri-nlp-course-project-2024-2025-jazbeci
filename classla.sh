#!/bin/bash
#SBATCH --job-name=gams
#SBATCH --output=logs/gams-%j.out
#SBATCH --error=logs/gams-%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00

export CLASSLA_RESOURCES_DIR=/d/hpc/home/ns97321/classla_resources

singularity exec ../containers/singularity_python3.10_classla.sif bash -c "
python finetuning_preprocessing_classla.py
"