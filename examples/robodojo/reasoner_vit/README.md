# Four-task joint14 WAM with current-image reasoner conditioning

The recipe is
`examples/toml/sft_config/action_policy_robodojo_four_tasks_wam_480_reasoner_vit.toml`.
It inherits the four-task joint-space 480/shift5 control: classify_objects,
insert_tubes, push_T and stack_bowls; raw absolute joint/gripper14 labels;
domain17; current state + 16 target actions; a 736×544 stitched camera canvas.
The optimizer, seed42, global batch2560, and 1600-step limit remain the same.
This recipe uses micro160 × accumulation1 ×16 GPUs, while the control uses
micro80 × accumulation2. This is an independent experiment/run name.
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

Additional ViT work and UND tokens increase memory and step time; micro160
requires a GPU capacity check. This implementation
is the **training path**. The current text-only policy server/evaluator must be
adapted to supply the same image+text UND inputs before evaluating this
checkpoint; this model rejects that old sampling path explicitly.
