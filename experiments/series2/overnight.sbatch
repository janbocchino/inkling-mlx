#!/bin/bash
#SBATCH --job-name=inkling2-overnight
#SBATCH --account=AIFAC_F02_743
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
set -Eeuo pipefail
module load python/3.11.7
cd "${SERIES_KIT:?}"
python -u controller.py --project "${SERIES_PROJECT:?}" --run "${SERIES_RUN:?}" --kit "$SERIES_KIT" &
controller_pid=$!
trap 'kill -USR1 "$controller_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$controller_pid" 2>/dev/null || true' TERM INT
wait "$controller_pid" || { status=$?; wait "$controller_pid"; exit "$status"; }
