"""Held-out normalized EEF20 validation, task-sharded across independent GPUs.

OmniMoTModel.validation_step is a stub in this revision. Evaluate the native
flow-matching objective through training_step under eval() and no_grad(). The
metrics describe noisy velocity prediction, not generated action MAE or success.
"""

import argparse
import gc
import hashlib
import json
import pickle
import time
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_results(output, checkpoint, plan):
    """Publish only complete, nonoverlapping task shards with paired settings."""
    tasks = {}
    parts = []
    for worker in range(plan["worker_count"]):
        path = output / "results" / f"{checkpoint.name}.worker_{worker}.json"
        part = json.loads(path.read_text())
        expected = {
            "checkpoint": str(checkpoint),
            "worker": worker,
            "seed": plan["noise_seed"],
            "batch_size": plan["batch_size"],
            "weights": "EMA",
            "smoke_only": False,
            "manifest_sha256": plan["manifest_sha256"],
            "normalization_sha256": plan["normalization_sha256"],
        }
        if any(part.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Validation shard settings disagree: {path}")
        for task, values in part["per_task"].items():
            if task in tasks or values["windows"] != plan["windows_by_task"].get(task):
                raise ValueError(f"Duplicate or incomplete validation task: {task}")
            tasks[task] = values
        parts.append(part)
    if set(tasks) != set(plan["windows_by_task"]):
        raise ValueError("Missing validation tasks")
    windows = sum(value["windows"] for value in tasks.values())
    if windows != plan["validation_windows"]:
        raise ValueError("Incomplete held-out coverage")
    result = {
        "step": int(checkpoint.name.removeprefix("iter_")),
        "checkpoint": str(checkpoint),
        "complete": True,
        "weights": "EMA",
        "plan": plan,
        "validation_windows": windows,
        "validation_episodes": plan["validation_episodes"],
        "per_task": tasks,
        "overall": {
            metric: sum(value[metric] * value["windows"] for value in tasks.values()) / windows
            for metric in plan["metrics"]
        },
        "task_macro_mean": {
            metric: sum(value[metric] for value in tasks.values()) / len(tasks) for metric in plan["metrics"]
        },
        "elapsed_seconds": max(part["elapsed_seconds"] for part in parts),
        "metric_definition": plan["metric_definition"],
    }
    atomic_json(output / "result.json", result)
    public = output.parent / "public"
    public.mkdir(parents=True, exist_ok=True)
    atomic_json(public / "validation.json", result)
    print("HELDOUT_COMPLETE " + json.dumps(result), flush=True)
    return result


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("wam", "policy"), default="wam")
    parser.add_argument(
        "--batch-index-offset",
        type=int,
        default=0,
        help="Preserve reference task-position noise seeds when evaluating a task subset",
    )
    parser.add_argument("--limit-batches", type=int, default=0, help="Smoke only; writes separate smoke results")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    if not (
        manifest.get("action_representation") == "eef_relative_chunk"
        and manifest.get("action_dim") == 20
        and manifest.get("domain_id") == 17
        and manifest.get("normalization_applied") is True
    ):
        raise ValueError("Expected the frozen normalized EEF20/domain17 manifest")
    tasks = validation_plan(manifest)
    task_names = sorted(tasks)
    if args.workers != len(task_names) or not 0 <= args.worker < args.workers:
        raise ValueError("Use exactly one independent GPU worker per held-out task")
    selected_tasks = {name: tasks[name] for name in task_names[args.worker :: args.workers]}
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
        "noise_seed_scheme": "seed + task_alphabetical_index*1000000 + task_batch_index",
        "metrics": ["total_loss", "action_flow_matching_loss", "vision_flow_matching_loss"],
        "metric_definition": "Native flow-matching validation objective; not rollout success or generated action reconstruction error. Weighted mean over all validation windows; task macro mean also provided.",
        "batch_size": args.batch_size,
        "worker_count": args.workers,
        "manifest_sha256": file_hash(args.prepared_root / "manifest.json"),
        "normalization_sha256": file_hash(args.prepared_root / "normalization.json"),
    }
    if args.worker == 0:
        atomic_json(args.output_dir / "plan.json", plan)
    if args.merge:
        for checkpoint in checkpoints:
            merge_results(args.output_dir, checkpoint, plan)
        return
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

    expected_root = config.dataloader_train.dataloader.datasets.robodojo.dataset.prepared_root
    if Path(expected_root).resolve() != args.prepared_root.resolve():
        raise ValueError("Validation data differs from the checkpoint's prepared root")
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
    # Every worker evaluates its fixed task for every pinned checkpoint.
    # No checkpoint-dependent noise seeds or data subsampling.
    assigned = checkpoints
    results_dir = args.output_dir / ("smoke_results" if args.limit_batches else "results")
    results_dir.mkdir(exist_ok=True)
    for checkpoint in assigned:
        result_path = results_dir / f"{checkpoint.name}.worker_{args.worker}.json"
        if result_path.is_file():
            continue
        started = time.monotonic()
        config.checkpoint.load_path = str(checkpoint)
        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        checkpointer.load(model)
        per_task = {}
        batch_index = 0
        for task, indices in selected_tasks.items():
            batch_index = task_names.index(task) * 1_000_000
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
                if args.limit_batches and windows >= args.limit_batches * args.batch_size:
                    break
            means = (sums / windows).tolist()
            per_task[task] = dict(zip(plan["metrics"], means), windows=windows)
            del loader
            gc.collect()
            torch.cuda.empty_cache()
            if args.limit_batches and windows >= args.limit_batches * args.batch_size:
                break
        total_windows = sum(v["windows"] for v in per_task.values())
        if not args.limit_batches and total_windows != sum(map(len, selected_tasks.values())):
            raise ValueError("Incomplete validation coverage")
        result = {
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name.removeprefix("iter_")),
            "worker": args.worker,
            "manifest_sha256": plan["manifest_sha256"],
            "normalization_sha256": plan["normalization_sha256"],
            "weights": "EMA",
            "seed": args.seed,
            "mode": args.mode,
            "noise_batch_index_offset": args.batch_index_offset,
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
