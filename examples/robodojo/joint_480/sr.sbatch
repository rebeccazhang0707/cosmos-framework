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
run=${1:?run root}
export ROBODOJO_SR_SEGMENT=${2:?segment number}
phase=${3:?phase}
python=/lustre/fsw/general_sa/rebeccaz/.venvs/hf-download/bin/python
tag=$(printf '%02d' "$ROBODOJO_SR_SEGMENT")
if ! /usr/bin/python3 -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["new_checkpoint"])' "$run/continuation/segment${tag}_finish.json"; then exit 0; fi
export ROBODOJO_SR_DEADLINE=$(($(date +%s) + 13800))
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$python" "$run/control/eval/sr.py" --initialize
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 "$python" "$run/control/eval/sr.py"
"$python" "$run/control/eval/sr.py" --finish "$phase"
