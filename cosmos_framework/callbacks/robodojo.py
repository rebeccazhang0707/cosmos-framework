"""Input checks, initialization records and deadline handling for RoboDojo."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils import misc
from cosmos_framework.utils.callback import Callback


class RoboDojoTrainingMonitor(Callback):
    def __init__(self, canvas_width=736, canvas_height=544):
        super().__init__()
        self.canvas_hw = (canvas_height, canvas_width)

    def on_train_start(self, model, iteration=0):
        self.root = Path(self.trainer.config.job.path_local) / "robodojo"
        self.root.mkdir(parents=True, exist_ok=True)
        self.deadline = float(os.environ.get("ROBODOJO_DEADLINE", "0"))
        self.checked = False
        self.iteration = None
        self.last_end = time.perf_counter()
        parameters = {}
        for name, parameter in model.named_parameters():
            if "net_ema" in name or not any(k in name for k in ("action2llm", "llm2action", "action_modality_embed")):
                continue
            value = parameter.detach()
            if hasattr(value, "to_local"):
                value = value.to_local()
            payload = value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            parameters[name] = dict(
                shape=list(parameter.shape),
                local_shape=list(value.shape),
                sha256=hashlib.sha256(payload).hexdigest(),
                trainable=parameter.requires_grad,
            )
        (self.root / f"init_rank_{dist.get_rank():02d}_step_{iteration}.json").write_text(
            json.dumps(dict(iteration=iteration, parameters=parameters), indent=2) + "\n"
        )
        reference = os.environ.get("ROBODOJO_REFERENCE_INIT_AUDIT")
        if iteration == 0 and reference:
            expected = json.loads((Path(reference) / f"init_rank_{dist.get_rank():02d}_step_0.json").read_text())
            if parameters != expected["parameters"]:
                raise ValueError("Initial action parameters differ from the paired reference")
            (self.root / f"init_match_rank_{dist.get_rank():02d}.json").write_text(
                json.dumps(dict(matched=True, reference=reference)) + "\n"
            )
        if iteration == 0:
            misc.set_random_seed(42, by_rank=True)

    def on_training_step_start(self, model, data, iteration=0):
        if not self.checked:

            def tensors(value):
                if isinstance(value, torch.Tensor):
                    yield value
                else:
                    for item in value:
                        yield from tensors(item)

            for plan in data["sequence_plan"]:
                if list(plan.condition_frame_indexes_vision) != [0] or list(plan.condition_frame_indexes_action) != [0]:
                    raise ValueError("RoboDojo conditions must contain only the current image and state")
            shapes = {tuple(clip.shape[-3:]) for clip in tensors(data[model.config.input_video_key])}
            rows = {int(action.shape[-2]) for action in tensors(data["action"])}
            if shapes != {(17, *self.canvas_hw)} or rows != {17}:
                raise ValueError(f"Unexpected WAM input layout: RGB{shapes}, action rows{rows}")
            (self.root / f"layout_rank_{dist.get_rank():02d}.json").write_text(
                json.dumps(dict(rgb_shapes=sorted(shapes), action_rows=17, samples=len(data["sequence_plan"]))) + "\n"
            )
            self.checked = True
        if iteration != self.iteration:
            torch.cuda.reset_peak_memory_stats()
            self.iteration = iteration

    def on_before_backward(self, model, loss, iteration=0):
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError("Nonfinite RoboDojo training loss")

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        now = time.perf_counter()
        free, total = torch.cuda.mem_get_info()
        row = dict(
            rank=dist.get_rank(),
            iteration=iteration,
            seconds=now - self.last_end,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            estimated_peak_used=torch.cuda.max_memory_reserved() + max(0, total - free - torch.cuda.memory_reserved()),
            total=total,
            loss=float(loss.detach()),
        )
        with (self.root / f"memory_rank_{dist.get_rank():02d}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.last_end = now
        if dist.get_rank() == 0:
            metrics = dict(step=iteration, total_loss=float(loss.detach()))
            for key in ("flow_matching_loss_action", "flow_matching_loss_vision"):
                metrics[key] = float(output_batch[key].detach())
            with (self.root / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if self.deadline:
            stop = torch.tensor(int(dist.get_rank() == 0 and time.time() >= self.deadline), device="cuda")
            dist.broadcast(stop, src=0)
            if int(stop):
                self.trainer.max_iterations = iteration
                if dist.get_rank() == 0:
                    (self.root / "budget_stop.json").write_text(json.dumps(dict(step=iteration)) + "\n")
