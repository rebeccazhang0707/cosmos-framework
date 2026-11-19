#!/bin/bash
# A transient filesystem/process failure must not silently lose the milestone.
set -uo pipefail
run=${1:?ViT run root}
base=${2:?baseline run root}
step=${3:-1000}
log="$run/control/evaluation_watch${step}.log"
scheduler="$run/control/eval/schedule_evaluation.py"
if [[ ! -f "$scheduler" ]]; then
    scheduler="$run/evaluation_runtime/examples/robodojo/reasoner_vit/schedule_evaluation.py"
fi
while true; do
    /usr/bin/python3 -u "$scheduler" \
        --run-root "$run" --base-run "$base" --step "$step" >>"$log" 2>&1
    result=$?
    if ((result == 0)) && /usr/bin/python3 -c \
        'import json,sys;sys.exit(not json.load(open(sys.argv[1])).get("job_id"))' \
        "$run/continuation/evaluation_schedule.json" >>"$log" 2>&1; then
        exit 0
    fi
    printf 'Watcher exited with code %s without a submitted job; retrying in60s\n' "$result" >>"$log"
    sleep 60
done
