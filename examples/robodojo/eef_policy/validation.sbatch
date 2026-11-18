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
val_run=${1:?run root}
val_repo=${2:?frozen validation source}
val_stages=${3:?comma-separated training segment numbers}
source "$val_run/environment.sh"
export PYTHONPATH="$val_repo"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cd "$val_repo"
val_report_python=${ROBODOJO_REPORT_PYTHON:-/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python}
IFS=',' read -ra val_stage_list <<< "$val_stages"
for val_stage in "${val_stage_list[@]}"; do
    val_tag=$(printf '%02d' "$val_stage")
    val_finish="$val_run/continuation/segment${val_tag}_finish.json"
    if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["new_checkpoint"])' "$val_finish"; then
        echo "Segment $val_stage has no new checkpoint; skip duplicate validation"
        continue
    fi
    val_train="$val_run/continuation/snapshots/segment${val_tag}"
    val_output="$val_run/continuation/evaluations/segment${val_tag}/validation"
    mkdir -p "$val_output/logs"
    val_args=(--train-dir "$val_train" --prepared-root "$val_run/prepared" --output-dir "$val_output" --workers 4 --batch-size 32 --seed 42 --mode wam)
    python -m cosmos_framework.evaluation.robodojo_eef_validation "${val_args[@]}" --plan-only
    val_pids=()
    for val_worker in 0 1 2 3; do
        env CUDA_VISIBLE_DEVICES="$val_worker" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 \
            MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000 + val_worker))" \
            python -u -m cosmos_framework.evaluation.robodojo_eef_validation "${val_args[@]}" --worker "$val_worker" \
            >"$val_output/logs/job${SLURM_JOB_ID}_worker${val_worker}.log" 2>&1 &
        val_pids+=("$!")
    done
    val_failed=0
    for val_pid in "${val_pids[@]}"; do
        if ! wait "$val_pid"; then val_failed=1; fi
    done
    if ((val_failed)); then
        echo "Validation worker failure for segment $val_stage; no complete point published"
        exit 1
    fi
    python -m cosmos_framework.evaluation.robodojo_eef_validation "${val_args[@]}" --merge
    "$val_report_python" -m cosmos_framework.evaluation.robodojo_eef_curves --run-root "$val_run"
done
