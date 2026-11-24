#!/bin/bash
#SBATCH --job-name=inkling2-matrices
#SBATCH --account=AIFAC_F02_743
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
set -Eeuo pipefail
module load profile/deeplrn
module load cineca-ai/4.3.0
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${SERIES_KIT:?}"
python -u matrix_service.py --run "${SERIES_RUN:?}" --seconds 42800
