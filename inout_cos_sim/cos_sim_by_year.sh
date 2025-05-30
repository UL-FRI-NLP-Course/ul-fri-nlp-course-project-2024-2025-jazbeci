  GNU nano 5.6.1          ./inout_cos_sim/cos_sim_by_year.sh          Modified  
#SBATCH --job-name=cos_sim_classla
#SBATCH --output=logs/classla-%j.out
#SBATCH --error=logs/classla-%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

export CLASSLA_RESOURCES_DIR=/d/hpc/home/ns97321/classla_resources

YEAR=$1

singularity exec ../../containers/singularity_python3.10_classla.sif bash -c "
python cosine_similarity.py $YEAR
"
# python lemmatize_classla.py $YEAR

# python pair.py

