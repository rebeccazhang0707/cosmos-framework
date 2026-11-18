#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --exclude=prenyx[0109-0110,0121-0122,0127]
#SBATCH --time=04:00:00
set -euo pipefail
run=${1:?run root}
stage=${2:?segment number}
source "$run/environment.sh"
cd "$ROBODOJO_REPO"
state=$(/usr/bin/python3 examples/robodojo/joint_480/segments.py prepare "$run" "$stage")
if [[ "$state" == COMPLETE ]]; then
    /usr/bin/python3 examples/robodojo/joint_480/segments.py finish "$run" "$stage"
    exit 0
fi
[[ "$SLURM_NNODES" == 2 ]]
export ROBODOJO_RESUME_FLAGS=''
if [[ "$state" == RESUME ]]; then
    export ROBODOJO_RESUME_FLAGS='checkpoint.load_training_state=True checkpoint.keys_to_skip_loading=[] checkpoint.strict_resume=True'
else
    [[ "$state" == FRESH ]]
fi
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
allocation_end=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
export ROBODOJO_DEADLINE="$(( $(date -d "$allocation_end" +%s) -120 ))"
export COSMOS_EXIT_WITHOUT_FINALIZE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$run/train/logs"
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 bash -c '
    exec python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
        --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        -m cosmos_framework.scripts.train \
        --sft-toml=examples/toml/sft_config/action_policy_robodojo_four_tasks_wam_480_shift5.toml -- \
        $ROBODOJO_RESUME_FLAGS
' >"$run/train/logs/train-${SLURM_JOB_ID}-segment${stage}.log" 2>&1
/usr/bin/python3 examples/robodojo/joint_480/segments.py finish "$run" "$stage"
