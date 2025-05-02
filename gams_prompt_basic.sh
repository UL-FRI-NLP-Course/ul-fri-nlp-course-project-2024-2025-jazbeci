#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=logs/test_model-%j.out
#SBATCH --error=logs/test_model-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=02:00:00

singularity exec --nv ../containers/container-pytorch2.6.0-transformers.sif bash -c "
export TRANSFORMERS_CACHE=/d/hpc/projects/onj_fri/jazbeci/cache_models && \
export CUDA_LAUNCH_BLOCKING=1 && \
python gams_prompt_basic.py
"
