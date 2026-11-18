# RoboDojo normalized relative EEF closed-loop evaluation

The model predicts 16 normalized EEF20 targets. Each arm is `[delta_xyz3,
delta_rotation6d, absolute_gripper1]`. All targets use the **same measured
chunk-start pose**: `p_target = delta_p + p_start`,
`R_target = delta_R @ R_start`. Rotation6D stores the first two matrix columns.
Quaternion transport uses wxyz. Coordinates are RoboDojo's environment-local
world frame; RobotManager performs its native robot-base/gripper transforms.

State and action use separate frozen training q01/q99 normalizers. State is the
current absolute EEF xyz/rotation6D/gripper, with no future GT inputs. The server
checks the checkpoint's EEF20/domain17/chunk16 contract and exact normalizer
bytes. Current images use the training camera composition and canvas padding.

The simulator accepts only a typed world-EEF packet. It calls
`RobotManager.solve_ik(..., trans="world")` for **each arm at each action**,
then passes the returned six-joint solutions and absolute grippers to the
native joint controller. EEF20 is never passed as joint targets. Native IK
failure holds the affected arm; grippers still apply. Nonfinite values and
invalid quaternions fail explicitly. Joint branch jumps are logged rather
than silently modifying the benchmark solver/controller.

`eef_native_ik.jsonl` beside each native result records starting-pose IK audits,
target EEF poses, resulting joints, IK failures, joint changes, and grip clipping.
An audit verifies that native IK can solve the measured starting poses.

Install this directory as a symlink named
`RoboDojo/XPolicyLab/policy/Cosmos3_Edge_RoboDojo_EEF`. Use a frozen Cosmos source
tree, keeping the training runtime unchanged. The Cosmos model and Isaac client
use separate Python/CUDA environments (`cell.sh`). The local Python/library
locations in the client launcher match the Prenyx installation.

`smoke.sbatch RUN INFERENCE_REPO ROBODOJO_REPO` runs a 32-step, one-episode
stack-bowls infrastructure check. It uses `RUN/continuation/smoke/deploy.yml`.
Its truncated episode is **not a success-rate estimate**. Formal periodic jobs
must also depend on successful smoke completion.

`periodic.sbatch RUN SEGMENT PHASE INFERENCE_REPO ROBODOJO_REPO` evaluates the
immutable `RUN/continuation/snapshots/segmentNN` saved at a training boundary.
The corresponding `segmentNN_finish.json` supplies `final_step` and
`new_checkpoint`. Future evaluation phases resume native episode manifests.
Each evaluation phase uses two nodes for at most four hours. The next training
segment depends only on preceding training, never on evaluation. Two evaluation
phases can be queued per snapshot; a completed quota costs no further inference.
No-op safety training segments skip duplicate evaluations.

The protocol matches the earlier joint experiment: 150 episodes per family,
600 total, seeds 0/1/2; stack-bowls and push-T include their standard/random
variants. Preserve five preselected episodes per family with all three native
camera videos. Video pruning affects only completed episodes in this unique
evaluation run. Results, episode counts, config/normalizer hashes and original
video links are published under `RUN/continuation/public/segmentNN_stepK/`.

## Paired held-out losses

Run `submit_validations.py --run-root RUN --repo FROZEN_VALIDATION_REPO
--source-commit COMMIT` after adding SR jobs to the continuation plan. It pairs
one validation job with each SR checkpoint, backfills finished checkpoints,
and skips duplicate SR phases and no-op safety segments. Each future job depends
only on the same completed training boundary as SR. Training and SR never wait
for validation, and the running training/SR source snapshots remain unchanged.

`validation.sbatch RUN FROZEN_VALIDATION_REPO SEGMENTS` uses four independent
GPU workers, one per task. It reads only immutable segment snapshots and all
held-out windows, with the checkpoint's official EEF videos, normalized EEF20
labels, current-state conditioning, camera canvas and training mode. It loads
EMA (matching SR), uses eval mode/no gradients, disables CFG dropout, fixes
per-task/per-batch noise seeds, and never updates weights. Native
`training_step` computes the validation objective because `validation_step`
is a stub in this framework revision. These are flow-matching losses rather
than generated-action reconstruction errors or rollout success.

Results contain window-weighted and task-macro action/image/total losses plus
per-task values and dataset/normalizer hashes. Incomplete shards cannot publish
a complete validation point. Completed results appear next to the matching SR
report as `validation.json`. After every checkpoint, the publisher automatically
refreshes `continuation/public/loss/` with training curves, EMA validation
points and downloadable per-task CSV data. The online training curve and EMA
validation curve use different weights, which is labeled on the plots.
