#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail
ulimit -c 0
export ROBODOJO_EEF_RUN=${1:?run root}
export ROBODOJO_EEF_STAGE=${2:?completed training segment}
eef_phase=${3:?evaluation phase}
export ROBODOJO_EEF_REPO=${4:?frozen inference repository}
export ROBODOJO_BENCH_ROOT=${5:?RoboDojo repository}
eef_python=${ROBODOJO_REPORT_PYTHON:-/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python}
eef_finish="$ROBODOJO_EEF_RUN/continuation/segment$(printf '%02d' "$ROBODOJO_EEF_STAGE")_finish.json"
# Extra safety segments that did not train must not duplicate the final evaluation.
if ! "$eef_python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["new_checkpoint"])' "$eef_finish"; then
    echo "No new checkpoint; skip duplicate evaluation"
    exit 0
fi
export ROBODOJO_SR_DEADLINE=$(($(date +%s) + 13800))
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROBODOJO_EEF_REPO"
cd "$ROBODOJO_EEF_REPO"
eef_script=cosmos_framework/evaluation/robodojo_eef_closed_loop.py
"$eef_python" "$eef_script" --initialize
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 "$eef_python" "$eef_script"
"$eef_python" "$eef_script" --finish "$eef_phase"
