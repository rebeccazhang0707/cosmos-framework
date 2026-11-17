#!/bin/bash
# Normalized relative EEF20 control; same 960x736/16-GPU recipe as joint exp1.
# Manual launch: same NNODES/NODE_RANK/MASTER_ADDR variables as the joint wrapper.
# Slurm: export ROBODOJO_ACTION_SPACE=eef, then sbatch examples/robodojo/four_tasks_wam.sbatch.
set -euo pipefail

TOML_FILE="examples/toml/sft_config/action_policy_robodojo_four_tasks_eef_wam.toml"
: "${ROBODOJO_PREPARED_ROOT:?Set the dedicated four-task EEF prepared directory}"
: "${BASE_CHECKPOINT_PATH:?Set the Cosmos3-Edge DCP directory}"
: "${WAN_VAE_PATH:?Set the Wan2.2_VAE.pth file}"
: "${NNODES:?Set NNODES=2 for the 16-GPU recipe}"
: "${NODE_RANK:?Set NODE_RANK=0 or 1}"
: "${MASTER_ADDR:?Set the address of node 0}"
if [[ "$NNODES" != 2 || "${NPROC_PER_NODE:-8}" != 8 ]]; then
    echo 'This recipe requires two nodes with eight GPUs each.' >&2
    exit 1
fi
export ROBODOJO_PREPARED_ROOT
if [[ "$ROBODOJO_PREPARED_ROOT" != /* ]]; then
    ROBODOJO_PREPARED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/$ROBODOJO_PREPARED_ROOT"
fi
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROBODOJO_PREPARED_ROOT"])
m = json.loads((root / "manifest.json").read_text())
expected = {"classify_objects", "insert_tubes", "push_T", "stack_bowls"}
for split, count in (("train", 360), ("val", 40)):
    episodes = [e for e in m["episodes"] if e["split"] == split]
    if len(episodes) != count or {e["task_name"] for e in episodes} != expected:
        raise ValueError(f"Expected the original four-task {split} split")
required = dict(action_dim=20, domain_id=17, action_representation="eef_relative_chunk",
                normalization_applied=True, chunk_length=16, sample_stride=8)
for key, value in required.items():
    if m.get(key) != value:
        raise ValueError(f"Wrong EEF contract: {key}={m.get(key)!r}, expected {value!r}")
stats = json.loads((root / m["normalization_file"]).read_text())
if stats["validation_used"] or stats["fit_split"] != "train" or stats["action"]["clip"] or stats["state"]["clip"]:
    raise ValueError("Expected frozen training-only state/action statistics without clipping")
PY
EXTRA_DATASET_CHECK='[[ -f "$ROBODOJO_PREPARED_ROOT/manifest.json" ]] || exit 1'
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
