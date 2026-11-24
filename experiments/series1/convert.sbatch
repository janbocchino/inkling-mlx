#!/bin/bash
#SBATCH --job-name=inkling-2bit
#SBATCH --account=AIFAC_F02_743
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/convert-%j.out
#SBATCH --error=logs/convert-%j.err

set -Eeuo pipefail
trap 'echo "Conversion job failed at line ${LINENO}" >&2' ERR
cd "${SLURM_SUBMIT_DIR:?Submit from the inkling-mlx project directory}"
project="$PWD"
test -d "$project/source"
# The venv's interpreter symlink resolves inside the container, not on the host.
test -f "$project/env-cpu/pyvenv.cfg"
test -f "$project/containers/python311-bookworm.sif"
mkdir -p "$project/converted"

# This uses CPU MLX, not the host's CUDA driver. No GPU is requested.
# The user's confirmed allocation is Booster; DCGP access is not assumed.
run_python() {
    srun --ntasks=1 singularity exec --cleanenv \
      --bind "$project:/work" --pwd /work \
      --env "HF_HOME=/work/hf-cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,MLX_ENABLE_TF32=0,OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,PYTHONUNBUFFERED=1" \
      "$project/containers/python311-bookworm.sif" \
      /work/env-cpu/bin/python -u "$@"
}

run_python /work/inkling-conversion-kit/test_converter.py
run_python /work/inkling-conversion-kit/convert_inkling.py \
  --source /work/source \
  --output "/work/converted/Inkling-2-Mini-expert-2bit-g64-${SLURM_JOB_ID}" \
  --group-size 64
