#!/bin/bash
set -euo pipefail
ulimit -c 0
output=$1
task=$2
seed=$3
model_gpu=$4
sim_gpu=$5
xpl=${ROBODOJO_BENCH_ROOT:?}/XPolicyLab
port=$(bash "$xpl/utils/get_free_port.sh")
cleanup() { if [[ -n ${server_pid:-} ]]; then kill "$server_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
bash "$ROBODOJO_EVAL_RUN/control/eval/server.sh" "$output" "$task" "$seed" "$model_gpu" "$port" &
server_pid=$!
bash "$xpl/utils/wait_for_policy_server.sh" localhost "$port" "$server_pid" "ViT image-conditioned WAM" 1200
bash "$xpl/policy/Cosmos3_Edge_RoboDojo_ViT/setup_eval_env_client.sh" \
    RoboDojo "$task" "step$ROBODOJO_EVAL_STEP" arx_x5 joint "$seed" "$sim_gpu" sim \
    "ckpt_name=step$ROBODOJO_EVAL_STEP,action_type=joint" "$port" localhost
