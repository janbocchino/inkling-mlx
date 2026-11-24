#!/bin/bash
#SBATCH --job-name=inkling2-source-pilot
#SBATCH --account=AIFAC_F02_743
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=00:45:00
set -Eeuo pipefail
cd "${SERIES_PROJECT:?}"
singularity exec --cleanenv --bind "$SERIES_PROJECT:/work,$SERIES_RUN:/run-series" --pwd /work/series2-kit \
  --env 'HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,MLX_ENABLE_TF32=0,OMP_NUM_THREADS=16,OPENBLAS_NUM_THREADS=16,PYTHONUNBUFFERED=1,SERIES_GPU_ENDPOINT=/run-series/GPU_ENDPOINT.json' \
  containers/python311-bookworm.sif /work/env-cpu/bin/python -u /work/series2-kit/test_rpc.py
singularity exec --cleanenv --bind "$SERIES_PROJECT:/work,$SERIES_RUN:/run-series" --pwd /work/series2-kit \
  --env 'HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,MLX_ENABLE_TF32=0,OMP_NUM_THREADS=16,OPENBLAS_NUM_THREADS=16,PYTHONUNBUFFERED=1,SERIES_GPU_ENDPOINT=/run-series/GPU_ENDPOINT.json' \
  containers/python311-bookworm.sif /work/env-cpu/bin/python -u \
  /work/series2-kit/cpu_phases.py pilot --run /run-series --seconds 2500
