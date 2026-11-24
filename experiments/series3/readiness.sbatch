#!/bin/bash
#SBATCH --job-name=inkling3-reference
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -Eeuo pipefail
# Explicit paths are supplied by the launcher after account/storage validation.
cd "${SERIES_PROJECT:?}"
singularity exec --cleanenv --bind "$SERIES_PROJECT:/work,$SERIES_RUN:/run-series" \
  --pwd /run-series/kit \
  --env "PYTHONUNBUFFERED=1,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,HF_HOME=/run-series/hf-cache" \
  containers/python311-bookworm.sif bash /run-series/kit/setup_reference.sh
