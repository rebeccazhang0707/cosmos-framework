#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=00:30:00
set -euo pipefail
ulimit -c 0
export ROBODOJO_EEF_RUN=${1:?run root}
export ROBODOJO_EEF_REPO=${2:?frozen inference repository}
export ROBODOJO_BENCH_ROOT=${3:?RoboDojo repository}
export ROBODOJO_RUN_ID="eef20_smoke_${SLURM_JOB_ID}"
export EVAL_NUM=1 EVAL_ENV_TYPE=sim ROBODOJO_EEF_SMOKE_STEPS=32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
exec bash "$ROBODOJO_EEF_REPO/examples/robodojo/eef_policy/cell.sh" \
    "$ROBODOJO_EEF_RUN/continuation/smoke" stack_bowls 0 0 1
