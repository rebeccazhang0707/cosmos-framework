# Joint exp1: 480 tier / shift5 control

The paired reference is `robodojo_edge_ab_20261109/.../exp1_edge`: Edge WAM,
raw absolute14D joint/gripper labels and clean joint state, four tasks,
360 training /40 held-out demos, domain17, seed42. This control reads the
**same joint-version videos and labels**, retaining micro80 × accumulation2
×16 GPUs = global batch2560, the exp1 optimizer/LR multipliers/warmup,
chunk16 / stride8, and original action-head initialization.

Changes are the 736×544 canvas, `resolution="480"`, and explicit shared RF
`shift=5`. The initial comparison stops at800 optimizer steps. Do not enlarge
the batch just because the smaller canvas frees memory.

`train.sbatch RUN SEGMENT` uses an isolated run's `environment.sh` and checks
its `control/config_audit.json`. Segment1 starts from Edge; subsequent segments
require all model/optimizer/scheduler/trainer DCP metadata. The native training
monitor stops after a full optimizer step before the four-hour deadline.
`segments.py` pins immutable evaluation snapshots and cancels unused future
segments after reaching800. The goal is not a reset at each Slurm boundary.

Each boundary has independent validation and native SR branches; training
never waits for either. SR targets150 episodes/task and keeps five native
three-camera bundles/task. Partial counts/video links publish every minute.

`validate.sbatch` evaluates all2222 held-out windows, EMA weights, with both
native `eval_shift=5` and reference-comparable `eval_shift=10`, using the same
batch size32 and seed42 protocol as the original joint exp1. Both results
record the evaluation override separately from the training shift. The latter
appears on the loss comparison curve; the former remains available as
`validation_native.json`. Geometry still differs, so image loss is not an
identical-target metric across the resolutions.
