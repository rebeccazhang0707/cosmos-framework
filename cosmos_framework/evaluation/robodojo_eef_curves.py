"""Publish native training losses and complete EMA held-out points together."""

import argparse
import csv
import fcntl
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

METRICS = (
    ("flow_matching_loss_action", "action_flow_matching_loss", "Action FM loss", "#2563eb"),
    ("flow_matching_loss_vision", "vision_flow_matching_loss", "Image / vision FM loss", "#059669"),
    ("total_loss", "total_loss", "Total loss", "#9333ea"),
)


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)


def publish(root, train_dir=None, label="EEF20 · four-task WAM"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = root / "continuation/public/loss"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".plot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if train_dir is None:
            train_dir = root / "train/robodojo/four_tasks_wam/edge_eef20_relative_960x736_16gpu"
        source = train_dir / "robodojo/training_metrics.jsonl"
        by_step = {}
        for line in source.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a live writer can leave its final line incomplete
            if not all(np.isfinite(row[key]) for key in ["step", *(m[0] for m in METRICS)]):
                raise ValueError("Nonfinite training metric")
            by_step[row["step"]] = row
        rows = [by_step[step] for step in sorted(by_step)]
        x = np.array([r["step"] for r in rows])
        if not len(x) or not np.all(np.diff(x) == 1):
            raise ValueError("Missing training steps")
        validations = []
        for path in sorted((root / "continuation/evaluations").glob("segment*/validation/result.json")):
            value = json.loads(path.read_text())
            if value["complete"]:
                validations.append(value)
        validations.sort(key=lambda v: v["step"])
        resumes = sorted(
            {
                json.loads(path.read_text())["resume_step"]
                for path in (root / "continuation").glob("segment*_start.json")
            }
        )
        window = 50

        def smooth(y):
            cumsum = np.r_[0.0, np.cumsum(y)]
            start = np.maximum(0, np.arange(len(y)) - window + 1)
            return (cumsum[np.arange(len(y)) + 1] - cumsum[start]) / (np.arange(len(y)) - start + 1)

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col", constrained_layout=True)
        for index, (train_key, val_key, label, color) in enumerate(METRICS):
            y = np.array([r[train_key] for r in rows])
            avg = smooth(y)
            for column in range(2):
                ax = axes[index, column]
                first = 1 if column == 0 else min(201, int(x[-1]))
                keep = x >= first
                ax.plot(x[keep], y[keep], lw=0.6, color=color, alpha=0.20, label="Train: per step")
                ax.plot(x[keep], avg[keep], lw=1.7, color=color, label="Train: trailing 50-step mean")
                points = [v for v in validations if v["step"] >= first]
                if points:
                    ax.plot(
                        [v["step"] for v in points],
                        [v["overall"][val_key] for v in points],
                        "o-",
                        color="#e76f00",
                        markersize=6,
                        lw=1.8,
                        label="Held-out: EMA, full split",
                    )
                for step in resumes:
                    if first <= step <= x[-1]:
                        ax.axvline(step, color="#64748b", ls="--", lw=0.8, alpha=0.6)
                if column == 0:
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(bottom=0)
                ax.set_xlim(first, max(first + 1, int(x[-1])))
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.18, which="both")
                if index == 0:
                    ax.set_title("Full history · logarithmic Y" if column == 0 else "Later training · linear Y")
                    ax.legend(frameon=False, fontsize=8)
                if index == 2:
                    ax.set_xlabel("Optimizer step")
        fig.suptitle(
            f"RoboDojo {label} · through step {x[-1]}\n"
            "Train = online weights; held-out = EMA used by SR, fixed noise seeds, no updates",
            fontsize=13,
        )
        for extension in ("png", "svg"):
            temp = out / f".loss_curves.tmp.{extension}"
            fig.savefig(temp, format=extension, dpi=160)
            os.replace(temp, out / f"loss_curves.{extension}")
        plt.close(fig)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["step", *(m[0] for m in METRICS)])
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(out / "training_losses.csv", stream.getvalue())
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["step", "task", "windows", *(m[1] for m in METRICS)])
        writer.writeheader()
        for value in validations:
            writer.writerow(
                {
                    "step": value["step"],
                    "task": "overall_window_weighted",
                    "windows": value["validation_windows"],
                    **value["overall"],
                }
            )
            for task, metrics in value["per_task"].items():
                writer.writerow({"step": value["step"], "task": task, **metrics})
        atomic_text(out / "validation_losses.csv", stream.getvalue())
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "max_step": int(x[-1]),
            "count": len(rows),
            "moving_average_steps": window,
            "resume_steps": resumes,
            "training_weights": "online",
            "validation_weights": "EMA (same as SR)",
            "label": label,
            "validation_checkpoints": [
                {
                    "step": v["step"],
                    "windows": v["validation_windows"],
                    "episodes": v["validation_episodes"],
                    "overall": v["overall"],
                    "evaluation_shift_override": v.get("evaluation_shift_override"),
                }
                for v in validations
            ],
            "metrics": {
                key: {"latest": rows[-1][key], "mean_last50": float(np.mean([r[key] for r in rows[-50:]]))}
                for key, *_ in METRICS
            },
            "definition": "Native flow-matching objective; validation covers the held-out split with fixed paired noise seeds",
        }
        atomic_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
        html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="120"><title>RoboDojo · train / held-out loss</title><style>body{font:16px system-ui;max-width:1360px;margin:30px auto;padding:0 20px;line-height:1.7;color:#172033}img{width:100%}a{color:#2563eb}td,th{padding:8px 15px;border-bottom:1px solid #ddd;text-align:left}aside{background:#f1f5f9;padding:12px 16px;border-radius:8px}</style><h1>RoboDojo · WAM 训练 / 验证 loss</h1>"""
        html += f'<p>{label}；训练记录至 step <b>{x[-1]}</b>；图表更新时间：{summary["generated_at_utc"]} UTC。<a href="../">闭环 SR 与视频</a>。</p>'
        html += "<aside>蓝/绿/紫：训练逐步值与向后50步均值；橙色点：完整 held-out split 的 EMA loss，与 SR 使用同一 checkpoint。验证固定噪声种子、关闭 CFG dropout，无梯度或参数更新。虚线：续训边界。指标是 flow-matching loss。</aside>"
        html += '<p>每次验证完成会自动重画此页；本页每2分钟刷新。训练曲线使用在线权重，验证使用 EMA，阅读两者差距时请注意。</p><p><a href="loss_curves.png">PNG</a> · <a href="loss_curves.svg">SVG</a> · <a href="training_losses.csv">训练 CSV</a> · <a href="validation_losses.csv">验证 CSV（含各任务）</a> · <a href="summary.json">统计与来源</a></p>'
        html += "<table><tr><th>Checkpoint</th><th>验证窗口</th><th>Action FM</th><th>Image FM</th><th>Total</th></tr>"
        for value in validations:
            v = value["overall"]
            html += f"<tr><td>{value['step']} · EMA</td><td>{value['validation_windows']}</td><td>{v['action_flow_matching_loss']:.6f}</td><td>{v['vision_flow_matching_loss']:.6f}</td><td>{v['total_loss']:.6f}</td></tr>"
        if not validations:
            html += '<tr><td colspan="5">验证任务已安排；完整结果完成后显示橙色点，待完成的数据不记为0。</td></tr>'
        html += f'</table><img src="loss_curves.png?v={summary["generated_at_utc"]}" alt="Train and held-out flow matching losses"></html>'
        atomic_text(out / "index.html", html)
        print(json.dumps(summary), flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--label", default="EEF20 · four-task WAM")
    args = parser.parse_args()
    publish(args.run_root, args.train_dir, args.label)
