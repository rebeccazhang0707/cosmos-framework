#!/bin/bash
set -euo pipefail
source "$ROBODOJO_EVAL_RUN/environment.sh"
export ROBODOJO_REPO="$ROBODOJO_EVAL_RUN/evaluation_runtime"
export PYTHONPATH="$ROBODOJO_REPO:$ROBODOJO_BENCH_ROOT:$ROBODOJO_BENCH_ROOT/XPolicyLab"
exec env CUDA_VISIBLE_DEVICES="$4" PYTHONWARNINGS=ignore::UserWarning \
    "$ROBODOJO_REPO/.venv/bin/python" "$ROBODOJO_BENCH_ROOT/XPolicyLab/setup_policy_server.py" \
    --config_path "$1/deploy.yml" --overrides port="$5" host=localhost bench_name=RoboDojo \
    task_name="$2" ckpt_name="step$ROBODOJO_EVAL_STEP" env_cfg_type=arx_x5 seed="$3" \
    policy_name=Cosmos3_Edge_RoboDojo_ViT action_type=joint
