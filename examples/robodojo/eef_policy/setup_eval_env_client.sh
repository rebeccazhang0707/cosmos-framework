#!/bin/bash
set -euo pipefail
bench_name=$1
task_name=$2
env_cfg_type=$4
seed=$6
env_gpu_id=$7
additional_info=$9
port=${10}
host=${11:-localhost}
XPL_ROOT="${ROBODOJO_BENCH_ROOT:?}/XPolicyLab"
BENCH_ROOT="$ROBODOJO_BENCH_ROOT"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/lustre/fsw/general_sa/rebeccaz/.cache/robodojo-graphics/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ${EVAL_ENV_TYPE:-sim} == debug ]]; then
    exec /lustre/fsw/general_sa/rebeccaz/Projects/cosmos-framework/.venv/bin/python \
        "${XPL_ROOT}/utils/debug_env_client.py" --bench_name "$bench_name" \
        --task_name "$task_name" --env_cfg_type "$env_cfg_type" \
        --policy_name Cosmos3_Edge_RoboDojo_EEF --protocol ws --host "$host" --port "$port" \
        --eval_batch false --eval_episode_num 1
fi
export PATH="/lustre/fsw/general_sa/rebeccaz/.venvs/robodojo-sim/bin:/lustre/fsw/general_sa/rebeccaz/.cache/ffmpeg-runtime/usr/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="/lustre/fsw/general_sa/rebeccaz/.cache/robodojo-graphics/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES=/lustre/fsw/general_sa/rebeccaz/.cache/robodojo-graphics/usr/share/vulkan/icd.d/nvidia_icd.json
robodojo_ffmpeg_lib=/lustre/fsw/general_sa/rebeccaz/.cache/ffmpeg-runtime/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$robodojo_ffmpeg_lib:$robodojo_ffmpeg_lib/pulseaudio:$robodojo_ffmpeg_lib/blas:$robodojo_ffmpeg_lib/lapack:/lustre/fsw/general_sa/rebeccaz/.cache/ffmpeg-extra-libs:$LD_LIBRARY_PATH"
# Keep CUDA and Vulkan physical device numbering aligned for IsaacLab.
unset CUDA_VISIBLE_DEVICES
cd "$BENCH_ROOT"
num_envs=$(python -c 'import sys; from XPolicyLab.utils.process_data import get_batch_size; print(get_batch_size(sys.argv[1]))' "$env_cfg_type")
if [[ ${EVAL_NUM:-native} =~ ^[0-9]+$ ]] && ((10#${EVAL_NUM} > 0 && 10#${EVAL_NUM} < num_envs)); then
    num_envs=$((10#${EVAL_NUM}))
fi
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$seed}"
robodojo_texture_cache="/lustre/fsw/general_sa/rebeccaz/.cache/robodojo-textures/${SLURM_JOB_ID:-local}_${env_gpu_id}"
mkdir -p "$robodojo_texture_cache"
exec python -u src/eval_client/main.py --task_name "$task_name" \
    --env_cfg_type "$env_cfg_type" --device_id "$env_gpu_id" --device "cuda:$env_gpu_id" \
    --num_envs "$num_envs" --enable_cameras --headless \
    --kit_args " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/rtx-transient/resourcemanager/localTextureCachePath=$robodojo_texture_cache" \
    --policy_name Cosmos3_Edge_RoboDojo_EEF --port "$port" --host "$host" --protocol ws \
    --policy_server_url "ws://$host:$port" --additional_info "$additional_info" --seed "$seed"
