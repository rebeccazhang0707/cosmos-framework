# Joint exp1: 480 tier / shift5 control

The paired reference is `robodojo_edge_ab_20261109/.../exp1_edge`: Edge WAM,
raw absolute14D joint/gripper labels and clean joint state, four tasks,
360 training /40 held-out demos, domain17, seed42. This control reads the
**same joint-version videos and labels**, retaining global batch2560
(micro160 × accumulation1 ×16 GPUs), the exp1 optimizer/LR multipliers/warmup,
chunk16 / stride8, and original action-head initialization.

Changes are the 736×544 canvas, `resolution="480"`, and explicit shared RF
`shift=5`. The initial comparison stops at1600 optimizer steps. Do not enlarge
the effective global batch just because the smaller canvas frees memory.
The first segment of the 20261117 control was already running micro80 ×
accumulation2 when this update was requested. It finishes without interruption;
segments2 onward resume its full checkpoint with micro160 × accumulation1.
The isolated continuation runtime differs only in these two TOML values;
the original runtime and model/optimizer schedule remain unchanged. Per-rank
memory/timing records track the measured capacity and speed after the switch.

`train.sbatch RUN SEGMENT` uses an isolated run's `environment.sh` and checks
its `control/config_audit.json`. Segment1 starts from Edge; subsequent segments
require all model/optimizer/scheduler/trainer DCP metadata. The native training
monitor stops after a full optimizer step before the four-hour deadline.
`segments.py` pins immutable evaluation snapshots and cancels unused future
segments after reaching1600. The goal is not a reset at each Slurm boundary.

Evaluate after **two four-hour training segments** (nominally eight hours),
plus the final goal checkpoint. Keep the already-running segment2 evaluation;
skip segments3 and5, then evaluate segments4 and6 and the final segment7 if
needed. Each evaluation has **one** `evaluate.sbatch RUN SEGMENT` job, requesting one
node / eight GPUs for at most four hours. It runs held-out validation first,
then SR on four policy/simulator GPU pairs. The existing eight logical SR
shards run two per pair sequentially, without sharing GPUs. Training never
waits for evaluation. SR targets15 episodes/task and keeps five native
three-camera bundles/task. Partial counts/video links publish every minute.
Do not submit separate validation and SR jobs for this recipe.

The combined job executes `validate.sbatch` as a shell helper inside its
allocation; the helper's SBATCH directives do not reserve another node.
It evaluates all2222 held-out windows, EMA weights, with both
native `eval_shift=5` and reference-comparable `eval_shift=10`, using the same
batch size32 and seed42 protocol as the original joint exp1. Both results
record the evaluation override separately from the training shift. The latter
appears on the loss comparison curve; the former remains available as
`validation_native.json`. Geometry still differs, so image loss is not an
identical-target metric across the resolutions.

SR protocol `sr15_seed3x5_v1` uses five valid episodes/family/seed,60 total.
Standard/random variants alternate3+2 and2+3 across seeds (8/7 overall).
Previously larger native results are summarized with an outcome-independent
first-quota prefix; the original outcomes remain unchanged and full reports are
archived under `legacy_150/`. All held-out loss windows remain in validation.
