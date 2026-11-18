#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail
run=${1:?run root}
stage=${2:?segment number}
source "$run/environment.sh"
cd "$ROBODOJO_REPO"
tag=$(printf '%02d' "$stage")
if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["new_checkpoint"])' "$run/continuation/segment${tag}_finish.json"; then exit 0; fi
snapshot="$run/continuation/snapshots/segment${tag}"
output="$run/continuation/evaluations/segment${tag}"
mkdir -p "$output/validation" "$output/validation_native"
pids=()
for shift in 5 10; do
    if [[ "$shift" == 5 ]]; then sub=validation_native; gpu=0; else sub=validation; gpu=1; fi
    env CUDA_VISIBLE_DEVICES="$gpu" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 \
        MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000 + gpu))" \
        python -u -m cosmos_framework.evaluation.robodojo_checkpoint_sweep \
        --train-dir "$snapshot" --output-dir "$output/$sub" --prepared-root "$ROBODOJO_PREPARED_ROOT" \
        --workers 1 --worker 0 --batch-size 32 --seed 42 --mode wam --eval-shift "$shift" \
        >"$output/$sub/job${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if ((failed)); then exit 1; fi
/usr/bin/python3 - "$output" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);(out/'public').mkdir(exist_ok=True)
for sub,name in [('validation','validation.json'),('validation_native','validation_native.json')]:
    paths=list((out/sub/'results').glob('iter_*.json'));assert len(paths)==1
    value=json.loads(paths[0].read_text());value['complete']=True
    assert value['validation_windows']==2222 and not value['smoke_only']
    (out/sub/'result.json').write_text(json.dumps(value,indent=2)+'\n')
    (out/'public'/name).write_text(json.dumps(value,indent=2)+'\n')
PY
/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python -m cosmos_framework.evaluation.robodojo_eef_curves \
    --run-root "$run" --train-dir "$run/train/robodojo/four_tasks_wam/edge_joint14_480_shift5_16gpu" \
    --label 'Joint14 · 480p / train shift5 · validation shift10'
