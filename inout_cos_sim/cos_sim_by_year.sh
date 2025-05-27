#!/bin/bash
#SBATCH --job-name=cos_sim_classla
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00

export CLASSLA_RESOURCES_DIR=/d/hpc/home/ns97321/classla_resources

YEAR=$1

LOG_OUT="logs/classla${YEAR}-%j.out"
LOG_ERR="logs/classla${YEAR}-%j.err"

#SBATCH --output=${LOG_OUT}
#SBATCH --error=${LOG_ERR}

singularity exec ../../containers/singularity_python3.10_classla.sif bash -c "
python pair.py $YEAR
"
# python lemmatize_classla.py $YEAR
# python cosine_similarity.py $YEAR