#!/bin/bash
# Run once on each of two 8-GPU nodes with a shared MASTER_ADDR and NODE_RANK=0/1.
# Raw joint14 WAM; 736x544 / shift5, current-image+text reasoner prefill, no AR.
set -euo pipefail

TOML_FILE="examples/toml/sft_config/action_policy_robodojo_four_tasks_wam_480_reasoner_vit.toml"
: "${ROBODOJO_PREPARED_ROOT:?Set the original four-task joint-action prepared directory}"
: "${BASE_CHECKPOINT_PATH:?Set the Cosmos3-Edge DCP directory}"
: "${WAN_VAE_PATH:?Set the Wan2.2_VAE.pth file}"
: "${NNODES:?Set NNODES=2 for the 16-GPU recipe}"
: "${NODE_RANK:?Set NODE_RANK=0 or 1}"
: "${MASTER_ADDR:?Set the address of node 0}"
if [[ "$NNODES" != 2 || "${NPROC_PER_NODE:-8}" != 8 ]]; then
    echo 'This recipe requires two nodes with eight GPUs each.' >&2
    exit 1
fi
: "${ROBODOJO_REASONER_CHECKPOINT_PATH:?Set the full local Cosmos3-Edge HF snapshot (with vision_encoder)}"
export ROBODOJO_REASONER_CHECKPOINT_PATH
if [[ "$ROBODOJO_REASONER_CHECKPOINT_PATH" != /* ]]; then
    ROBODOJO_REASONER_CHECKPOINT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/$ROBODOJO_REASONER_CHECKPOINT_PATH"
fi
[[ -f "$ROBODOJO_REASONER_CHECKPOINT_PATH/vision_encoder/model.safetensors" ]] || {
    echo 'Missing Edge vision encoder weights.' >&2
    exit 1
}
export ROBODOJO_PREPARED_ROOT
# Resolve a relative data directory from the repository root, like the shared launcher.
if [[ "$ROBODOJO_PREPARED_ROOT" != /* ]]; then
    ROBODOJO_PREPARED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/$ROBODOJO_PREPARED_ROOT"
fi
python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads((Path(os.environ["ROBODOJO_PREPARED_ROOT"]) / "manifest.json").read_text())
expected = {"classify_objects", "insert_tubes", "push_T", "stack_bowls"}
for split in ("train", "val"):
    tasks = {e["task_name"] for e in manifest["episodes"] if e["split"] == split}
    if tasks != expected:
        raise ValueError(f"Expected the original four tasks in {split}, got {sorted(tasks)}")
if manifest.get("action_dim", 14) != 14 or manifest.get("action_representation", "joint_absolute") != "joint_absolute":
    raise ValueError("This recipe requires raw absolute 14D joint/gripper labels")
if manifest.get("domain_id", 17) != 17:
    raise ValueError("The original four-task recipe uses domain_id=17")
PY
EXTRA_DATASET_CHECK='[[ -f "$ROBODOJO_PREPARED_ROOT/manifest.json" ]] || { echo "Missing RoboDojo manifest.json" >&2; exit 1; }'
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
