# Four-task joint14 WAM with current-image reasoner conditioning

The recipe is
`examples/toml/sft_config/action_policy_robodojo_four_tasks_wam_480_reasoner_vit.toml`.
It inherits the four-task joint-space 480/shift5 control: classify_objects,
insert_tubes, push_T and stack_bowls; raw absolute joint/gripper14 labels;
domain17; current state + 16 target actions; a 736×544 stitched camera canvas.
The optimizer, seed42, global batch2560, and 1600-step limit remain the same.
This recipe uses micro160 × accumulation1 ×16 GPUs. The control now uses the
same layout from its second segment; its first segment used micro80 ×
accumulation2. This is an independent experiment/run name.
Full-block activation checkpointing recomputes attention during backward to
reduce the measured micro160 peak; the paired launcher enables expandable
allocator segments. The original selective checkpoint policy ran out of memory
during the first backward with micro160 and image-conditioned UND tokens.

Each training sample takes **only RGB t=0**, after the existing resize/padding,
through the released, frozen Edge SigLIP2 encoder and projector. The native
Edge processor produces 391 merged image tokens at this canvas size. Image
tokens precede the existing serialized JSON task caption in the user message;
dataset text dropout is preserved, while the current image remains available.
Native image/text mRoPE is computed separately for each packed sample.

The projected image features replace image placeholders on the causal UND
stream. The frozen reasoner processes that stream together with text, producing
layerwise K/V consumed by GEN through the existing MoT attention. UND cannot
attend to GEN or another sample. The original VAE observation/action-state
conditioning and flow-matching targets remain in place. Only the usual GEN and
action parameters are optimized. There is no reasoner answer loss, autoregressive
decode, generated plan, or persistent KV cache; training computes the prefill
as part of the joint forward for each new observation.

Set these shared paths on both nodes:

```bash
export ROBODOJO_PREPARED_ROOT=/path/to/original/four_task_joint_dataset
export BASE_CHECKPOINT_PATH=/path/to/base/edge_dcp
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export ROBODOJO_REASONER_CHECKPOINT_PATH=/path/to/full/Cosmos3-Edge
```

The reasoner path must be a local Edge HF snapshot with tokenizer/processor
metadata, `config.json`, and `vision_encoder/model.safetensors`. It must match
the Edge version used to initialize the DCP. This path supplies the frozen ViT
and projector; the existing base DCP supplies the reasoner language weights.
ViT is loaded outside FSDP, remains frozen/eval, and is not copied into every
training checkpoint. Retain the same snapshot for resume.

For two nodes with eight GPUs each, set `NNODES=2`, `NODE_RANK=0` or `1`, and a
shared `MASTER_ADDR`, then run on each node:

```bash
bash examples/launch_sft_action_policy_robodojo_four_tasks_wam_480_reasoner_vit.sh
```

Additional ViT work and UND tokens increase memory and step time. The current
model also supports WAM policy sampling with `guidance=1`: it calls the same
current-image ViT/processor as training, attaches features before the first
denoising pack, and retains those features for every denoising step. Sampling
rejects text CFG, negative prompts and prompt rewriting. Only current RGB and
joint state condition the prediction; there is no cross-request visual cache.

The first evaluation targets the exact saved checkpoint1000 (about the fifth
four-hour segment at the measured63.5 seconds/step). A CPU-only watcher calls
`schedule_evaluation.py`: it waits for all model/optimizer/scheduler/trainer DCP
metadata, pins immutable config/checkpoint links, then submits one GPU job.
It inserts that job into the baseline evaluation dependency chain so at most
one evaluation node runs across both experiments. No GPU waits for training
to reach the target. Native SR and full held-out losses share one
one-node/eight-GPU job and retain five three-camera bundles/task.
An additional checkpoint can use `--status-file` without replacing the
step1000 milestone. `--after-eval JOB` preserves an already-queued baseline
evaluation's position; the milestone waits while an earlier ViT evaluation
is pending or running. The supervisor uses a pinned controller under
`control/eval/` when present, keeping model inference code frozen separately.
Use an isolated inference runtime containing the image-conditioned sampler;
leave the frozen running training runtime unchanged.
