#!/bin/bash
# Submit from repo root, with the cu130 environment active and paths exported:
# ROBODOJO_PREPARED_ROOT, BASE_CHECKPOINT_PATH, WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT.
# ROBODOJO_ACTION_SPACE=joint selects raw14D exp1; =eef selects normalized relative20D.
# Both use 960x736, global batch2560, 16 GPUs, LR5e-5 and warmup50.
# The EEF prepared manifest must preserve the joint control's demo order/splits.
# Example: ROBODOJO_ACTION_SPACE=eef sbatch examples/robodojo/four_tasks_wam.sbatch
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --exclude=prenyx[0109-0110,0121-0122,0127]
#SBATCH --time=04:00:00
set -euo pipefail
[[ "$SLURM_NNODES" == 2 ]] || { echo 'Expected2 nodes/16 GPUs' >&2; exit 1; }
: "${IMAGINAIRE_OUTPUT_ROOT:?Set a fresh persistent output root}"
export ROBODOJO_REPO="${ROBODOJO_REPO:-$SLURM_SUBMIT_DIR}"
cd "$ROBODOJO_REPO"
case "${ROBODOJO_ACTION_SPACE:-joint}" in
    joint) export ROBODOJO_LAUNCHER=examples/launch_sft_action_policy_robodojo_four_tasks_wam.sh ;;
    eef) export ROBODOJO_LAUNCHER=examples/launch_sft_action_policy_robodojo_four_tasks_eef_wam.sh ;;
    *) echo 'ROBODOJO_ACTION_SPACE must be joint or eef' >&2; exit 1 ;;
esac
export PATH="$ROBODOJO_REPO/.venv/bin:$PATH"
export PYTHONPATH="$ROBODOJO_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export COSMOS_EXIT_WITHOUT_FINALIZE=1
export NNODES=2 NPROC_PER_NODE=8
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
# Derive the deadline from Slurm's allocation end, including prolog/startup time.
# Stop after a complete optimizer step; measured final saves take about7 seconds.
allocation_end=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
export ROBODOJO_DEADLINE="$(( $(date -d "$allocation_end" +%s) - ${ROBODOJO_END_MARGIN_SECONDS:-120} ))"
mkdir -p "$OUTPUT_ROOT/logs"
printf 'Training %s; deadline epoch %s; end margin %ss\n' "${ROBODOJO_ACTION_SPACE:-joint}" "$ROBODOJO_DEADLINE" "${ROBODOJO_END_MARGIN_SECONDS:-120}"
srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 bash -c '
    export NODE_RANK="$SLURM_PROCID"
    export LOG_FILENAME="train-${SLURM_JOB_ID}-node-${SLURM_PROCID}.log"
    exec bash "$ROBODOJO_LAUNCHER"
'
