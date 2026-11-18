#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=04:00:00
# One allocation: held-out losses, then four policy/simulator GPU pairs.
set -euo pipefail
ulimit -c 0
run=${1:?run root}
stage=${2:?segment number}
source "$run/environment.sh"
tag=$(printf '%02d' "$stage")
finish="$run/continuation/segment${tag}_finish.json"
if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["new_checkpoint"])' "$finish"; then exit 0; fi

# This helper runs on GPUs 0/1 inside this allocation; bash ignores its SBATCH headers.
validation_failed=0
if ! bash "$run/control/eval/validate.sbatch" "$run" "$stage"; then
    validation_failed=1
fi

export ROBODOJO_SR_SEGMENT="$stage"
export ROBODOJO_SR_STEP
ROBODOJO_SR_STEP=$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["final_step"])' "$finish")
allocation_end=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
export ROBODOJO_SR_DEADLINE="$(( $(date -d "$allocation_end" +%s) -180 ))"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
report_python=/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python
sr_script="$run/control/eval/sr.py"
output="$run/continuation/evaluations/segment${tag}"
"$report_python" "$sr_script" --initialize

# The existing eight logical shards use pair index%4. Execute shards i and i+4
# sequentially in each lane, preserving the protocol and avoiding GPU sharing.
pids=()
for lane in 0 1 2 3; do
    (
        for shard in "$lane" "$((lane + 4))"; do
            "$report_python" "$sr_script" --worker "$shard"
        done
    ) >"$output/logs/combined_lane${lane}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done
sr_failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then sr_failed=1; fi
done
if ! "$report_python" "$sr_script" --finish 1; then sr_failed=1; fi
if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$output/public/report.json"; then
    sr_failed=1
fi

/usr/bin/python3 - "$output" "$validation_failed" "$sr_failed" <<'PY'
import json, os, sys
from pathlib import Path
out = Path(sys.argv[1])
value = dict(job_id=os.environ['SLURM_JOB_ID'], nodes=1, gpus=8,
             validation_ok=sys.argv[2] == '0', sr_ok=sys.argv[3] == '0')
temporary = out / 'combined_status.tmp'
temporary.write_text(json.dumps(value, indent=2) + '\n')
temporary.replace(out / 'combined_status.json')
PY
if ((validation_failed || sr_failed)); then exit 1; fi
