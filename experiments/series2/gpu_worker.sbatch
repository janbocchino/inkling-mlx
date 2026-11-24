#!/bin/bash
#SBATCH --job-name=inkling2-search
#SBATCH --account=AIFAC_F02_743
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -Eeuo pipefail
module load profile/deeplrn
module load cineca-ai/4.3.0
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${SERIES_KIT:?}"
if [[ "${PARITY_ONLY:-0}" == 1 ]]; then
    python -u gpu_worker.py --run "${SERIES_RUN:?}" --seconds 600 --parity-only
else
    python -u gpu_worker.py --run "${SERIES_RUN:?}" --seconds 10600
fi
