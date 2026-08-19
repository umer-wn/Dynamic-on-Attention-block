#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHECKPOINTS = ["step1000", "step41000", "step81000", "step121000"]
COLORS = {
    "step1000": "#2878b5",
    "step41000": "#dc8a2e",
    "step81000": "#2f855a",
    "step121000": "#6b4c9a",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_inline(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("|", "\\|")
        .replace("`", "\\`")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-report-dir")
    args = parser.parse_args()
    root = Path(args.root)
    figures = root / "figures"
    processed = root / "processed"
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, list[dict]] = {}
    summaries: list[dict] = []
    for checkpoint in CHECKPOINTS:
        raw = root / checkpoint / "raw"
        trajectory_paths = sorted(raw.glob("*__trajectory.jsonl"))
        summary_paths = sorted(raw.glob("*__summary.jsonl"))
        if len(trajectory_paths) != 1 or len(summary_paths) != 1:
            raise RuntimeError(
                f"{checkpoint}: expected one trajectory and summary, "
                f"got {len(trajectory_paths)} and {len(summary_paths)}"
            )
        rows = read_jsonl(trajectory_paths[0])
        summary_rows = read_jsonl(summary_paths[0])
        if len(rows) != 256 or len(summary_rows) != 1:
            raise RuntimeError(
                f"{checkpoint}: expected 256 trajectory rows and one summary, "
                f"got {len(rows)} and {len(summary_rows)}"
            )
        trajectories[checkpoint] = rows
        summaries.extend(summary_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for ax, checkpoint in zip(axes.flat, CHECKPOINTS):
        rows = trajectories[checkpoint][-128:]
        x = np.asarray([float(row["projection_0"]) for row in rows])
        y = np.asarray([float(row["projection_1"]) for row in rows])
        steps = np.asarray([int(row["text_step"]) for row in rows])
        ax.plot(x, y, color=COLORS[checkpoint], lw=1.1, alpha=0.55)
        points = ax.scatter(
            x,
            y,
            c=steps,
            cmap="viridis",
            s=30,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.scatter(
            x[0],
            y[0],
            marker="o",
            s=70,
            color="#111111",
            facecolor="none",
            linewidth=1.3,
            label="step129",
        )
        ax.scatter(
            x[-1],
            y[-1],
            marker="X",
            s=70,
            color="#d62728",
            linewidth=0.8,
            label="step256",
        )
        summary = next(row for row in summaries if row["checkpoint"] == checkpoint)
        cycle = (
            "no repeated window"
            if summary["text_cycle_length"] is None
            else (
                f"cycle start={summary['text_cycle_start']}, "
                f"length={summary['text_cycle_length']}"
            )
        )
        ax.set_title(f"{checkpoint} (training step {summary['training_step']:,})\n{cycle}")
        ax.set_xlabel("full 8-token embedding projection 0")
        ax.set_ylabel("full 8-token embedding projection 1")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
        fig.colorbar(points, ax=ax, label="text-level dynamic step")
    fig.suptitle(
        "Greedy 8-token text-level dynamics — final 128 of 256 steps\n"
        "Each text step = 8 sequential next-token generations; fixed projections",
        fontsize=15,
    )
    figure_path = figures / "full_text_token_projection_last128.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_table = [
        {
            "checkpoint": row["checkpoint"],
            "training_step": row["training_step"],
            "initial_text": row["initial_text"],
            "text_steps": row["text_steps"],
            "tokens_per_text_step": row["tokens_per_text_step"],
            "total_generated_tokens": row["total_generated_tokens"],
            "unique_text_windows": row["unique_text_windows"],
            "text_cycle_start": row["text_cycle_start"],
            "text_cycle_length": row["text_cycle_length"],
            "mean_token_change_fraction": row["mean_token_change_fraction"],
            "final_text": row["final_text"],
            "runtime_seconds": row["runtime_seconds"],
        }
        for row in sorted(summaries, key=lambda item: int(item["training_step"]))
    ]
    write_csv(processed / "checkpoint_summary.csv", summary_table)

    initial_texts = {row["initial_text"] for row in summaries}
    initial_ids = {tuple(row["initial_token_ids"]) for row in summaries}
    projection_hashes = {row["projection_vectors_sha256"] for row in summaries}
    if len(initial_texts) != 1 or len(initial_ids) != 1 or len(projection_hashes) != 1:
        raise RuntimeError(
            "checkpoints must share one initial token window and one projection basis"
        )
    table_lines = "\n".join(
        (
            f"| {row['checkpoint']} | {row['unique_text_windows']} | "
            f"{row['text_cycle_start']} | {row['text_cycle_length']} | "
            f"{row['mean_token_change_fraction']:.3f} | "
            f"`{markdown_inline(row['final_text'])}` |"
        )
        for row in summary_table
    )
    readme = f"""# Full-text token projection dynamics

状态：`complete`

## 定义

- 模型：`EleutherAI/pythia-70m`
- checkpoints：`step1000`、`step41000`、`step81000`、`step121000`，严格间隔 40,000 training steps。
- 初始状态：同一个本地 OpenWebMath 样本的前 8 个 token：`{next(iter(initial_texts))}`。
- 每个微步按正常因果推理执行一次 greedy argmax，只生成 1 个 token。
- 微步之后丢弃窗口中最老的 token，并把新 token 加到末尾；不 padding、不使用 KV cache，position id 每次重置为 `0..7`。
- 连续执行 8 个微步后，窗口正好由 8 个新 token 构成，记为一个 text-level dynamic step。
- 共执行 256 个 text-level steps，即每个 checkpoint 实际生成 2,048 个 token。
- 投影状态是当前完整 8-token 窗口对应的 `8×H` input embedding，展平后使用跨 checkpoint 固定的随机单位向量投影。
- 图中仅使用最后 128 个 text-level steps，即 step129–step256。

## 数据流

```text
8-token window S_t
  -> generate one token and shift window, repeated 8 times
  -> new 8-token window S_(t+1)
  -> embedding lookup E(S_(t+1)) in R^(8×H)
  -> fixed 2-D projection (visualization only)
```

生成第 `j` 个新 token 时，模型看到的是旧窗口尚未移出的后缀以及本轮已经生成的新 token 前缀。
到第 8 个微步结束时，上一轮的 8 个 token 已全部移出。

## 最后 128 步投影

![最后 128 个 text-level steps 的全文 token 投影](figures/full_text_token_projection_last128.png)

## 摘要

| checkpoint | unique windows | cycle start | cycle length | mean position change | final text |
|---|---:|---:|---:|---:|---|
{table_lines}

`cycle` 按完整 8-token 离散窗口是否重复判定。投影重合本身不用于判定周期。

## 产物

- `processed/checkpoint_summary.csv`
- `figures/full_text_token_projection_last128.png`
- 各 checkpoint 下的 `raw/*__trajectory.jsonl`
- 各 checkpoint 下的 `raw/*__summary.jsonl`

运行命令见 `RUNBOOK.md`。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    if args.repo_report_dir:
        repo = Path(args.repo_report_dir)
        (repo / "figures").mkdir(parents=True, exist_ok=True)
        (repo / "processed").mkdir(parents=True, exist_ok=True)
        (repo / "figures" / figure_path.name).write_bytes(figure_path.read_bytes())
        (repo / "processed" / "checkpoint_summary.csv").write_bytes(
            (processed / "checkpoint_summary.csv").read_bytes()
        )
        (repo / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "checkpoints": len(CHECKPOINTS),
                "trajectory_rows": sum(len(rows) for rows in trajectories.values()),
                "plotted_rows": 128 * len(CHECKPOINTS),
                "figure": str(figure_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
