#!/bin/bash
set -euo pipefail
ulimit -c 0
eef_out=$1
eef_task=$2
eef_seed=$3
eef_model_gpu=$4
eef_sim_gpu=$5
eef_repo=${ROBODOJO_EEF_REPO:?}
eef_xpl=${ROBODOJO_BENCH_ROOT:?}/XPolicyLab
eef_port=$(bash "$eef_xpl/utils/get_free_port.sh")
cleanup() { if [[ -n ${eef_server_pid:-} ]]; then kill "$eef_server_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
(
    source "$ROBODOJO_EEF_RUN/environment.sh"
    export PYTHONPATH="$eef_repo:$ROBODOJO_BENCH_ROOT:$eef_xpl"
    export CUDA_VISIBLE_DEVICES=$eef_model_gpu
    exec python -u "$eef_xpl/setup_policy_server.py" --config_path "$eef_out/deploy.yml" --overrides \
        port="$eef_port" host=localhost bench_name=RoboDojo task_name="$eef_task" \
        ckpt_name=eef env_cfg_type=arx_x5 seed="$eef_seed" \
        policy_name=Cosmos3_Edge_RoboDojo_EEF action_type=ee
) &
eef_server_pid=$!
bash "$eef_xpl/utils/wait_for_policy_server.sh" localhost "$eef_port" "$eef_server_pid" "EEF policy server" 1200
# CUDA13 model libraries are isolated in the subshell above, never inherited by Isaac.
export PYTHONPATH="$eef_repo${PYTHONPATH:+:$PYTHONPATH}"
bash "$eef_repo/examples/robodojo/eef_policy/setup_eval_env_client.sh" \
    RoboDojo "$eef_task" eef arx_x5 ee "$eef_seed" "$eef_sim_gpu" sim \
    "ckpt_name=eef,action_type=ee" "$eef_port" localhost
