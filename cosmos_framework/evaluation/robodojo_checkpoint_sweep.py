"""Offline validation of saved RoboDojo checkpoints, with no optimizer updates.

OmniMoTModel.validation_step is a stub in this revision. Evaluate the native
flow-matching objective through training_step under eval() and no_grad(). The
metrics describe noisy velocity prediction, not generated action MAE or success.
"""

import argparse
import gc
import json
import pickle
import time
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validation_plan(manifest):
    tasks = {}
    offset = 0
    for episode in manifest["episodes"]:
        if episode["split"] != "val":
            continue
        count = max(0, (episode["length"] - 17) // 8 + 1)
        tasks.setdefault(episode["task_name"], []).extend(range(offset, offset + count))
        offset += count
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-shift", type=int, help="Optional shared RF noise shift for cross-resolution validation")
    parser.add_argument("--mode", choices=("wam", "policy"), default="wam")
    parser.add_argument(
        "--batch-index-offset",
        type=int,
        default=0,
        help="Preserve reference task-position noise seeds when evaluating a task subset",
    )
    parser.add_argument("--limit-batches", type=int, default=0, help="Smoke only; writes separate smoke results")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    tasks = validation_plan(manifest)
    checkpoints = sorted(
        p for p in (args.train_dir / "checkpoints").glob("iter_*") if (p / "model/.metadata").is_file()
    )
    if not checkpoints:
        raise ValueError("No complete model checkpoints found")
    plan = {
        "checkpoints": [str(p) for p in checkpoints],
        "checkpoint_count": len(checkpoints),
        "validation_episodes": sum(e["split"] == "val" for e in manifest["episodes"]),
        "validation_windows": sum(map(len, tasks.values())),
        "windows_by_task": {k: len(v) for k, v in tasks.items()},
        "chunk_length": 16,
        "sample_stride": 8,
        "cfg_dropout_rate": 0.0,
        "weights": "EMA",
        "mode": args.mode,
        "noise_seed": args.seed,
        "noise_batch_index_offset": args.batch_index_offset,
        "evaluation_shift_override": args.eval_shift,
        "metrics": ["total_loss", "action_flow_matching_loss", "vision_flow_matching_loss"],
        "metric_definition": "Native flow-matching validation objective; not rollout success or generated joint MAE. Weighted mean over all validation windows; task macro mean also provided.",
        "batch_size": args.batch_size,
        "worker_count": args.workers,
    }
    if args.worker == 0:
        atomic_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    import attrs
    import torch
    from torch.utils.data import DataLoader, Subset

    from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.datasets.robodojo_policy_dataset import (
        RoboDojoPolicyDataset,
        RoboDojoTransform,
    )
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.lazy_config import instantiate

    distributed.init()
    misc.set_random_seed(args.seed)
    with (args.train_dir / "config.pkl").open("rb") as f:
        config = pickle.load(f)

    def thaw(obj):
        if attrs.has(type(obj)):
            if hasattr(obj, "_is_frozen"):
                obj._is_frozen = False
            for field in attrs.fields(type(obj)):
                thaw(getattr(obj, field.name))

    thaw(config)
    config.job.name = f"checkpoint_validation_worker_{args.worker}"
    config.job.group = "offline_validation"
    m = config.model.config
    training_shift = m.rectified_flow_training_config.shift
    if args.eval_shift is not None:
        m.rectified_flow_training_config.shift = args.eval_shift
    m.parallelism.data_parallel_shard_degree = 1
    m.parallelism.data_parallel_replicate_degree = 1
    m.parallelism.context_parallel_shard_degree = 1
    m.parallelism.cfg_parallel_shard_degree = 1
    m.activation_checkpointing.mode = "none"
    m.compile.enabled = False
    # Keep EMA present so native DCP loading can copy saved EMA into net.
    m.ema.enabled = True
    config.checkpoint.load_training_state = False
    config.checkpoint.load_ema_to_reg = True
    config.checkpoint.keys_to_skip_loading = []
    config.checkpoint.strict_resume = True
    config.checkpoint.dcp_async_mode_enabled = False
    model = instantiate(config.model)
    model.eval()

    raw = RoboDojoPolicyDataset(str(args.prepared_root), split="val", chunk_length=16, sample_stride=8, mode=args.mode)
    if len(raw) != plan["validation_windows"]:
        raise ValueError("Validation window plan disagrees with dataset")
    dataset_config = config.dataloader_train.dataloader.datasets.robodojo.dataset
    canvas_width = int(dataset_config.get("canvas_width", 960))
    canvas_height = int(dataset_config.get("canvas_height", 736))
    resolution = str(dataset_config.get("resolution", "720"))
    if resolution != str(m.resolution):
        raise ValueError("Dataset and model resolution tiers disagree")
    plan.update(resolution=resolution, canvas_wh=[canvas_width, canvas_height])
    if args.worker == 0:
        atomic_json(args.output_dir / "plan.json", plan)
    transform = RoboDojoTransform(
        tokenizer_config=m.vlm_config.tokenizer,
        max_action_dim=m.max_action_dim,
        cfg_dropout_rate=0.0,
        format_prompt_as_json=True,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    dataset = ActionSFTDataset(raw, transform, resolution)
    # Workers run independent one-GPU process groups. Every checkpoint sees
    # identical batches and RNG seeds regardless of its assigned worker.
    assigned = checkpoints[args.worker :: args.workers]
    if args.limit_batches:
        assigned = assigned[:1]
    if args.worker == (len(checkpoints) - 1) % args.workers:
        assigned = list(reversed(assigned))  # expose the final model result early
    results_dir = args.output_dir / ("smoke_results" if args.limit_batches else "results")
    results_dir.mkdir(exist_ok=True)
    for checkpoint in assigned:
        result_path = results_dir / f"{checkpoint.name}.json"
        if result_path.is_file():
            continue
        started = time.monotonic()
        config.checkpoint.load_path = str(checkpoint)
        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        checkpointer.load(model)
        per_task = {}
        batch_index = 0
        for task, indices in tasks.items():
            inner_loader = DataLoader(
                Subset(dataset, indices),
                batch_size=1,
                shuffle=False,
                num_workers=2,
                prefetch_factor=1,
                collate_fn=custom_collate_fn,
                pin_memory=False,
            )
            loader = PackingDataLoader(
                dataloader=inner_loader,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                patch_spatial=2,
                max_sequence_length=None,
                max_samples_per_batch=args.batch_size,
                sound_latent_fps=0,
                dataset_name="robodojo_validation",
            )
            sums = torch.zeros(3, dtype=torch.float64, device="cuda")
            windows = 0
            for batch in loader:
                # No dependence on checkpoint iteration. Re-seed before VAE,
                # sigma and noise draws to make checkpoint comparisons paired.
                misc.set_random_seed(args.seed + args.batch_index_offset + batch_index)
                batch = misc.to(batch, device="cuda")
                count = len(batch["sequence_plan"])
                with torch.no_grad():
                    output, loss = model.training_step(batch, iteration=0)
                values = torch.stack(
                    [
                        loss.detach().double(),
                        output["flow_matching_loss_action"].detach().double(),
                        output["flow_matching_loss_vision"].detach().double(),
                    ]
                ).reshape(3)
                if not torch.isfinite(values).all():
                    raise ValueError(f"Non-finite validation metric: {checkpoint.name}, {task}, {batch_index}")
                sums += values * count
                windows += count
                batch_index += 1
                atomic_json(
                    args.output_dir / f"worker_{args.worker}_progress.json",
                    {
                        "checkpoint": checkpoint.name,
                        "task": task,
                        "windows_in_task": windows,
                        "task_total": len(indices),
                        "batch": batch_index,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                del output, loss, values, batch
                if args.limit_batches and batch_index >= args.limit_batches:
                    break
            means = (sums / windows).tolist()
            per_task[task] = dict(zip(plan["metrics"], means), windows=windows)
            del loader
            gc.collect()
            torch.cuda.empty_cache()
            if args.limit_batches and batch_index >= args.limit_batches:
                break
        total_windows = sum(v["windows"] for v in per_task.values())
        if not args.limit_batches and total_windows != plan["validation_windows"]:
            raise ValueError("Incomplete validation coverage")
        result = {
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name.removeprefix("iter_")),
            "weights": "EMA",
            "seed": args.seed,
            "mode": args.mode,
            "noise_batch_index_offset": args.batch_index_offset,
            "evaluation_shift_override": args.eval_shift,
            "training_shift": dict(training_shift) if not isinstance(training_shift, int) else training_shift,
            "batch_size": args.batch_size,
            "validation_windows": total_windows,
            "validation_episodes": plan["validation_episodes"],
            "per_task": per_task,
            "overall": {
                key: sum(v[key] * v["windows"] for v in per_task.values()) / total_windows for key in plan["metrics"]
            },
            "task_macro_mean": {key: sum(v[key] for v in per_task.values()) / len(per_task) for key in plan["metrics"]},
            "elapsed_seconds": time.monotonic() - started,
            "smoke_only": bool(args.limit_batches),
            "metric_definition": plan["metric_definition"],
        }
        atomic_json(result_path, result)
        print("VALIDATION_RESULT " + json.dumps(result), flush=True)
        del checkpointer
        gc.collect()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
