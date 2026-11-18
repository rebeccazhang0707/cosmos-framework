#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail
ulimit -c 0
run=${1:?run root}
step=${2:?checkpoint step}
# Model CUDA libraries are sourced only by validation/server subprocesses,
# keeping them out of the Isaac client's parent environment.
export ROBODOJO_EVAL_RUN="$run" ROBODOJO_EVAL_STEP="$step" ROBODOJO_SR_STEP="$step"
export ROBODOJO_EVAL_REPO="$run/evaluation_runtime"
export ROBODOJO_BENCH_ROOT=/lustre/fsw/general_sa/rebeccaz/Projects/RoboDojo
export PYTHONPATH="$ROBODOJO_EVAL_REPO"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
report_python=/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python
sr_script="$ROBODOJO_EVAL_REPO/cosmos_framework/evaluation/robodojo_vit_closed_loop.py"
validation_failed=0
if ! bash "$run/control/eval/validate.sbatch" "$run" "$step"; then validation_failed=1; fi
allocation_end=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
export ROBODOJO_SR_DEADLINE="$(( $(date -d "$allocation_end" +%s) -180 ))"
"$report_python" "$sr_script" --initialize
output="$run/continuation/evaluations/step$(printf '%06d' "$step")"
pids=()
for lane in 0 1 2 3; do
    (
        for shard in "$lane" "$((lane + 4))"; do
            "$report_python" "$sr_script" --worker "$shard"
        done
    ) >"$output/logs/lane${lane}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done
sr_failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then sr_failed=1; fi; done
if ! "$report_python" "$sr_script" --finish 1; then sr_failed=1; fi
if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$output/public/report.json"; then sr_failed=1; fi
if ((validation_failed || sr_failed)); then exit 1; fi
