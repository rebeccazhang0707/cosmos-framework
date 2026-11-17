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
bash "$ROBODOJO_EEF_REPO/examples/robodojo/eef_policy/cell.sh" \
    "$ROBODOJO_EEF_RUN/continuation/smoke" stack_bowls 0 0 1
# Benchmark exception recovery must not turn a partial smoke into a successful gate.
/usr/bin/python3 - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["ROBODOJO_BENCH_ROOT"]) / "eval_result/RoboDojo/stack_bowls/Cosmos3_Edge_RoboDojo_EEF/arx_x5"
paths = list(base.glob(f'*/{os.environ["ROBODOJO_RUN_ID"]}/eef_native_ik.jsonl'))
assert len(paths) == 1, paths
rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
actions = [row for row in rows if "action_index" in row]
assert len(actions) == 32, f"Expected32 executed EEF actions, got{len(actions)}"
assert {row["chunk"] for row in actions} == {0, 1}, "Must test a second observed chunk anchor"
print(json.dumps({"smoke_passed": True, "eef_actions": len(actions), "native_ik_arm_calls": 2 * len(rows)}))
PY
