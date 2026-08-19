#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


CHECKPOINTS = ["step0", "step1000", "step16000", "step143000"]
FREQUENCY_BINS = 8
TOKENS_PER_BIN = 4
COLORS = {
    "step0": "#9c3f35",
    "step1000": "#dc8a2e",
    "step16000": "#2878b5",
    "step143000": "#2f855a",
}
METRIC_LABELS = {
    "lyapunov_mean": "Conditional Lyapunov exponent",
    "tail_relative_step_delta_mean": "Tail relative step delta",
    "hutchinson_frobenius_median": "Normalized Frobenius (median)",
    "nearby_log_growth_per_step": "Nearby log growth / step",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def checkpoint_step(value: str) -> int:
    return int(value.removeprefix("step"))


def load_frequency_rows(single_root: Path) -> tuple[list[dict], list[dict]]:
    trajectory_rows: list[dict] = []
    summary_rows: list[dict] = []
    for checkpoint in CHECKPOINTS:
        shard_root = single_root / "pilot" / checkpoint
        trajectory_paths = sorted(shard_root.glob("shard*/raw/*__trajectory.jsonl"))
        summary_paths = sorted(shard_root.glob("shard*/raw/*__summary.jsonl"))
        if len(trajectory_paths) != 4 or len(summary_paths) != 4:
            raise RuntimeError(
                f"{checkpoint}: expected four trajectory and summary shards, "
                f"got {len(trajectory_paths)} and {len(summary_paths)}"
            )
        for path in trajectory_paths:
            trajectory_rows.extend(
                row
                for row in read_jsonl(path)
                if row.get("group") == "isolated_token" and int(row["step"]) >= 512
            )
        for path in summary_paths:
            for row in read_jsonl(path):
                if row.get("group") != "isolated_token":
                    continue
                frobenius = [finite(value) for value in row.get("hutchinson_normalized_frobenius", [])]
                frobenius = [value for value in frobenius if value is not None]
                row["hutchinson_frobenius_median"] = (
                    float(np.median(frobenius)) if frobenius else None
                )
                summary_rows.append(row)
    keys = {
        (row["checkpoint"], int(row["token_id"]), int(row["context_id"]))
        for row in summary_rows
    }
    expected = len(CHECKPOINTS) * FREQUENCY_BINS * TOKENS_PER_BIN
    if len(summary_rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"expected {expected} unique isolated-token summaries, "
            f"got {len(summary_rows)}/{len(keys)}"
        )
    return trajectory_rows, summary_rows


def bin_descriptions(summary_rows: list[dict]) -> dict[int, str]:
    by_bin: dict[int, list[int]] = defaultdict(list)
    for row in summary_rows:
        if row["checkpoint"] == "step0":
            by_bin[int(row["frequency_bin"])].append(int(row["frequency_count"]))
    return {
        bin_id: (
            f"Bin {bin_id} (rank Q{bin_id + 1}/{FREQUENCY_BINS}, "
            f"count {min(values)}–{max(values)})"
        )
        for bin_id, values in sorted(by_bin.items())
    }


def projection_figures(
    trajectory_rows: list[dict],
    summary_rows: list[dict],
    descriptions: dict[int, str],
    figures: Path,
) -> None:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    token_meta: dict[int, tuple[str, int]] = {}
    for row in trajectory_rows:
        grouped[(row["checkpoint"], int(row["frequency_bin"]), int(row["token_id"]))].append(row)
        token_meta[int(row["token_id"])] = (str(row["decoded"]), int(row["frequency_count"]))
    cmap = plt.get_cmap("viridis")
    for checkpoint in CHECKPOINTS:
        fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
        for bin_id in range(FREQUENCY_BINS):
            ax = axes.flat[bin_id]
            token_ids = sorted(
                {
                    int(row["token_id"])
                    for row in summary_rows
                    if row["checkpoint"] == checkpoint and int(row["frequency_bin"]) == bin_id
                },
                key=lambda token_id: token_meta[token_id][1],
            )
            for token_index, token_id in enumerate(token_ids):
                rows = sorted(grouped[(checkpoint, bin_id, token_id)], key=lambda row: int(row["step"]))
                x = np.array([float(row["projection_0"]) for row in rows])
                y = np.array([float(row["projection_1"]) for row in rows])
                decoded, count = token_meta[token_id]
                color = cmap(0.12 + 0.78 * token_index / max(len(token_ids) - 1, 1))
                ax.plot(x, y, lw=1.15, alpha=0.88, color=color, label=f"{decoded!r} (n={count})")
                ax.scatter(x[0], y[0], marker="o", s=34, color=color, edgecolor="black", linewidth=0.7, zorder=4)
                ax.scatter(x[-1], y[-1], marker="X", s=46, color=color, edgecolor="black", linewidth=0.7, zorder=5)
                if checkpoint == "step16000":
                    ax.annotate(
                        "S512",
                        (x[0], y[0]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6,
                        color=color,
                    )
                    if token_index == 0:
                        ax.annotate(
                            "E768 cluster",
                            (x[-1], y[-1]),
                            xytext=(4, -10),
                            textcoords="offset points",
                            fontsize=6,
                            color="#222222",
                        )
            ax.set_title(descriptions[bin_id], fontsize=10, fontweight="bold")
            ax.set_ylabel("projection 1")
            ax.set_xlabel("projection 0")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="best", framealpha=0.82)
        fig.suptitle(
            f"Single-token feedback trajectories by token frequency — {checkpoint}\n"
            "Pythia-70M; isolated token; steps 512–768; ○ S512, × E768",
            fontsize=15,
        )
        fig.savefig(
            figures / f"single_token_frequency_projection_{checkpoint}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def metric_figure(summary_rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items()):
        for checkpoint in CHECKPOINTS:
            rows = [row for row in summary_rows if row["checkpoint"] == checkpoint]
            x, y = [], []
            for row in rows:
                value = finite(row.get(metric))
                if value is None:
                    continue
                if metric == "nearby_log_growth_per_step" and bool(row.get("nearby_numerical_floor")):
                    continue
                x.append(math.log10(int(row["frequency_count"]) + 1))
                y.append(value)
            ax.scatter(x, y, s=35, alpha=0.72, color=COLORS[checkpoint], label=checkpoint)
            if len(x) >= 3:
                slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), deg=1)
                xx = np.linspace(min(x), max(x), 100)
                ax.plot(xx, slope * xx + intercept, color=COLORS[checkpoint], lw=1.2, alpha=0.9)
        if metric in {"tail_relative_step_delta_mean", "hutchinson_frobenius_median"}:
            ax.set_yscale("log")
        ax.axhline(0.0, color="#555555", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("log10(WikiText-2 token count + 1)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    count_per_checkpoint = len(summary_rows) // len(CHECKPOINTS)
    fig.suptitle(
        f"Frequency–dynamics relationships ({count_per_checkpoint} tokens per checkpoint)",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def frequency_tables(summary_rows: list[dict], processed: Path) -> tuple[list[dict], list[dict]]:
    correlations: list[dict] = []
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = [row for row in summary_rows if row["checkpoint"] == checkpoint]
        for metric in METRIC_LABELS:
            pairs: list[tuple[float, float]] = []
            excluded_floor = 0
            for row in checkpoint_rows:
                if metric == "nearby_log_growth_per_step" and bool(row.get("nearby_numerical_floor")):
                    excluded_floor += 1
                    continue
                value = finite(row.get(metric))
                if value is None:
                    continue
                pairs.append((math.log10(int(row["frequency_count"]) + 1), value))
            if len(pairs) >= 3:
                result = spearmanr([pair[0] for pair in pairs], [pair[1] for pair in pairs])
                rho, pvalue = float(result.statistic), float(result.pvalue)
            else:
                rho, pvalue = float("nan"), float("nan")
            correlations.append(
                {
                    "checkpoint": checkpoint,
                    "training_step": checkpoint_step(checkpoint),
                    "metric": metric,
                    "n": len(pairs),
                    "excluded_numerical_floor": excluded_floor,
                    "spearman_rho": rho,
                    "p_value": pvalue,
                }
            )

    bins: list[dict] = []
    for checkpoint in CHECKPOINTS:
        for bin_id in range(FREQUENCY_BINS):
            rows = [
                row
                for row in summary_rows
                if row["checkpoint"] == checkpoint and int(row["frequency_bin"]) == bin_id
            ]
            output = {
                "checkpoint": checkpoint,
                "training_step": checkpoint_step(checkpoint),
                "frequency_bin": bin_id,
                "n_tokens": len(rows),
                "frequency_count_min": min(int(row["frequency_count"]) for row in rows),
                "frequency_count_max": max(int(row["frequency_count"]) for row in rows),
                "tokens": " | ".join(
                    f"{row['decoded']!r}:{int(row['frequency_count'])}"
                    for row in sorted(rows, key=lambda item: int(item["frequency_count"]))
                ),
            }
            for metric in METRIC_LABELS:
                values = [finite(row.get(metric)) for row in rows]
                values = [value for value in values if value is not None]
                output[f"{metric}_median"] = float(np.median(values)) if values else None
            bins.append(output)
    write_csv(processed / "frequency_metric_correlations.csv", correlations)
    write_csv(processed / "frequency_bin_summary.csv", bins)
    write_csv(
        processed / "isolated_token_summary.csv",
        [
            {
                "checkpoint": row["checkpoint"],
                "training_step": checkpoint_step(row["checkpoint"]),
                "frequency_bin": int(row["frequency_bin"]),
                "token_id": int(row["token_id"]),
                "token": row["decoded"],
                "frequency_count": int(row["frequency_count"]),
                **{metric: row.get(metric) for metric in METRIC_LABELS},
                "nearby_numerical_floor": bool(row.get("nearby_numerical_floor")),
            }
            for row in summary_rows
        ],
    )
    return correlations, bins


def load_loss_completions(root: Path, source_id: str) -> dict[int, dict]:
    output: dict[int, dict] = {}
    for path in sorted((root / "raw" / source_id).glob("step*/loss_complete.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "complete":
            output[int(row["training_step"])] = row
    return output


def loss_figures(loss_rows: list[dict], figures: Path) -> None:
    specs = [
        ("train_loss_by_checkpoint.png", "train_loss", "The Pile train-split proxy loss"),
        ("test_loss_by_checkpoint.png", "test_loss", "The Pile test loss"),
    ]
    for filename, column, title in specs:
        fig, (ax, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8.5), constrained_layout=True)
        x = [row["training_step"] for row in loss_rows]
        y = [row[column] for row in loss_rows]
        ax.plot(x, y, color="#2878b5", lw=2.0, marker="o", ms=4)
        ax.set_ylabel("Token-weighted causal cross-entropy")
        ax.set_title(f"{title}: full range")
        ax.grid(alpha=0.25)
        zoom_rows = [row for row in loss_rows if row["training_step"] >= 1000]
        ax_zoom.plot(
            [row["training_step"] for row in zoom_rows],
            [row[column] for row in zoom_rows],
            color="#2878b5",
            lw=2.0,
            marker="o",
            ms=4,
        )
        ax_zoom.set_xlabel("Training checkpoint step")
        ax_zoom.set_ylabel("Token-weighted causal cross-entropy")
        ax_zoom.set_title("Zoom: step1000 and later")
        ax_zoom.grid(alpha=0.25)
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, (ax_loss, ax_gap) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    x = [row["training_step"] for row in loss_rows]
    train = [row["train_loss"] for row in loss_rows]
    test = [row["test_loss"] for row in loss_rows]
    gap = [row["test_minus_train"] for row in loss_rows]
    ax_loss.plot(x, train, color="#2878b5", lw=2, marker="o", ms=4, label="train-split proxy")
    ax_loss.plot(x, test, color="#9c3f35", lw=2, marker="o", ms=4, label="test")
    ax_loss.set_ylabel("Token-weighted loss")
    ax_loss.set_title("The Pile train/test loss across Pythia-70M checkpoints")
    ax_loss.legend()
    ax_loss.grid(alpha=0.25)
    ax_gap.plot(x, gap, color="#6b4c9a", lw=2, marker="o", ms=4)
    ax_gap.axhline(0.0, color="#555555", lw=0.8, ls="--")
    ax_gap.set_xlabel("Training checkpoint step")
    ax_gap.set_ylabel("test − train corpus difference")
    ax_gap.grid(alpha=0.25)
    fig.savefig(figures / "train_test_loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_loss_table(train_root: Path, test_root: Path, processed: Path, figures: Path) -> list[dict]:
    train = load_loss_completions(train_root, "the_pile_train")
    test = load_loss_completions(test_root, "the_pile_test")
    shared = sorted(set(train) & set(test))
    if len(shared) < 20:
        raise RuntimeError(f"expected broad train/test checkpoint overlap, got {len(shared)}")
    rows = [
        {
            "checkpoint": f"step{step}",
            "training_step": step,
            "train_loss": float(train[step]["token_weighted_loss"]),
            "test_loss": float(test[step]["token_weighted_loss"]),
            "test_minus_train": float(test[step]["token_weighted_loss"])
            - float(train[step]["token_weighted_loss"]),
            "train_predicted_tokens": int(train[step]["predicted_token_count"]),
            "test_predicted_tokens": int(test[step]["predicted_token_count"]),
        }
        for step in shared
    ]
    write_csv(processed / "checkpoint_train_test_loss.csv", rows)
    loss_figures(rows, figures)
    return rows


def hard_loss_figures(loss_rows: list[dict], figures: Path) -> None:
    fig, (ax, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8.5), constrained_layout=True)
    x = [row["training_step"] for row in loss_rows]
    hard = [row["hard_natural_language_loss"] for row in loss_rows]
    ax.plot(x, hard, color="#6b4c9a", lw=2.0, marker="o", ms=4)
    ax.set_ylabel("Token-weighted causal cross-entropy")
    ax.set_title("Local hard natural-language loss: full range")
    ax.grid(alpha=0.25)
    zoom_rows = [row for row in loss_rows if row["training_step"] >= 1000]
    ax_zoom.plot(
        [row["training_step"] for row in zoom_rows],
        [row["hard_natural_language_loss"] for row in zoom_rows],
        color="#6b4c9a",
        lw=2.0,
        marker="o",
        ms=4,
    )
    ax_zoom.set_xlabel("Training checkpoint step")
    ax_zoom.set_ylabel("Token-weighted causal cross-entropy")
    ax_zoom.set_title("Zoom: step1000 and later")
    ax_zoom.grid(alpha=0.25)
    fig.savefig(figures / "local_hard_natural_language_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    corpus_specs = [
        ("train_loss", "The Pile train proxy", "#2878b5"),
        ("test_loss", "The Pile test", "#9c3f35"),
        ("hard_natural_language_loss", "local OpenWebMath hard set", "#6b4c9a"),
    ]
    axis_specs = [
        (False, False, "linear x / linear y", "three_corpus_loss_linear_x_linear_y.png"),
        (True, False, "log x / linear y", "three_corpus_loss_log_x_linear_y.png"),
        (False, True, "linear x / log y", "three_corpus_loss_linear_x_log_y.png"),
        (True, True, "log x / log y", "three_corpus_loss_log_x_log_y.png"),
    ]

    def draw_loss_axes(ax, log_x: bool, log_y: bool, title: str) -> None:
        plot_x = [step + 1 for step in x] if log_x else x
        for column, label, color in corpus_specs:
            ax.plot(
                plot_x,
                [row[column] for row in loss_rows],
                color=color,
                lw=2,
                marker="o",
                ms=4,
                label=label,
            )
        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Training checkpoint step + 1 (log scale)")
        else:
            ax.set_xlabel("Training checkpoint step")
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel("Token-weighted loss")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    for log_x, log_y, title, filename in axis_specs:
        fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
        draw_loss_axes(ax, log_x, log_y, title)
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for ax, (log_x, log_y, title, _) in zip(axes.flat, axis_specs):
        draw_loss_axes(ax, log_x, log_y, title)
    fig.suptitle("Three-corpus loss: all linear/log axis combinations", fontsize=16)
    fig.savefig(figures / "three_corpus_loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax_difference = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    ax_difference.plot(
        x,
        [row["hard_minus_test"] for row in loss_rows],
        color="#6b4c9a",
        lw=2,
        marker="o",
        ms=4,
        label="hard − The Pile test",
    )
    ax_difference.plot(
        x,
        [row["hard_minus_train"] for row in loss_rows],
        color="#dc8a2e",
        lw=2,
        marker="o",
        ms=4,
        label="hard − The Pile train proxy",
    )
    ax_difference.plot(
        x,
        [row["test_minus_train"] for row in loss_rows],
        color="#2f855a",
        lw=2,
        marker="o",
        ms=4,
        label="The Pile test − train proxy",
    )
    ax_difference.axhline(0.0, color="#555555", lw=0.8, ls="--")
    ax_difference.set_xlabel("Training checkpoint step")
    ax_difference.set_ylabel("Corpus loss difference")
    ax_difference.legend()
    ax_difference.grid(alpha=0.25)
    fig.savefig(figures / "three_corpus_loss_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def extend_with_hard_loss(
    loss_rows: list[dict],
    hard_root: Path,
    processed: Path,
    figures: Path,
) -> list[dict]:
    hard = load_loss_completions(hard_root, "open_web_math_local_hard")
    expected_steps = {int(row["training_step"]) for row in loss_rows}
    missing = sorted(expected_steps - set(hard))
    if missing:
        raise RuntimeError(f"missing local hard-corpus loss checkpoints: {missing}")
    for row in loss_rows:
        step = int(row["training_step"])
        hard_loss = float(hard[step]["token_weighted_loss"])
        row["hard_natural_language_loss"] = hard_loss
        row["hard_minus_test"] = hard_loss - float(row["test_loss"])
        row["hard_minus_train"] = hard_loss - float(row["train_loss"])
        row["hard_predicted_tokens"] = int(hard[step]["predicted_token_count"])
    write_csv(processed / "checkpoint_three_corpus_loss.csv", loss_rows)
    hard_loss_figures(loss_rows, figures)
    return loss_rows


def format_correlation_rows(rows: list[dict], metric: str) -> str:
    selected = [row for row in rows if row["metric"] == metric]
    return "\n".join(
        f"| {row['checkpoint']} | {row['n']} | {row['spearman_rho']:.3f} | {row['p_value']:.3g} |"
        for row in selected
    )


def write_readme(
    output_root: Path,
    descriptions: dict[int, str],
    correlations: list[dict],
    bin_rows: list[dict],
    loss_rows: list[dict],
    train_metadata: dict,
    hard_metadata: dict,
) -> None:
    tokens_by_bin = []
    for bin_id in range(FREQUENCY_BINS):
        row = next(
            item for item in bin_rows if item["checkpoint"] == "step0" and item["frequency_bin"] == bin_id
        )
        tokens_by_bin.append(
            f"| {bin_id} | {descriptions[bin_id]} | {row['tokens']} |"
        )
    final = loss_rows[-1]
    min_test = min(loss_rows, key=lambda row: row["test_loss"])
    min_hard = min(loss_rows, key=lambda row: row["hard_natural_language_loss"])
    hard_above_test = sum(
        row["hard_natural_language_loss"] > row["test_loss"] for row in loss_rows
    )
    hard_above_train = sum(
        row["hard_natural_language_loss"] > row["train_loss"] for row in loss_rows
    )
    lyap = [row for row in correlations if row["metric"] == "lyapunov_mean"]
    strongest = max(lyap, key=lambda row: abs(row["spearman_rho"]) if math.isfinite(row["spearman_rho"]) else -1)
    final_frobenius = next(
        row
        for row in correlations
        if row["checkpoint"] == "step143000"
        and row["metric"] == "hutchinson_frobenius_median"
    )
    step0_nearby = next(
        row
        for row in correlations
        if row["checkpoint"] == "step0" and row["metric"] == "nearby_log_growth_per_step"
    )
    step1000_nearby = next(
        row
        for row in correlations
        if row["checkpoint"] == "step1000" and row["metric"] == "nearby_log_growth_per_step"
    )
    readme = f"""# 单 token 词频动力学与三语料 Loss

状态：`complete`

## 研究问题

本实验回答两个问题：

1. 同一 Pythia-70M checkpoint 下，不同词频 token 进行单 token 循环
   `x_(t+1)=Transformer(x_t)[-1]` 时，投影轨迹与局部动力学指标是否存在稳定差异？
2. Pythia-70M 随 checkpoint 训练时，The Pile train/test loss 如何变化？
3. 换成服务器本地已有、数学内容更密集的自然语言语料后，loss 曲线是否更难？

## 数据与方法

- 模型：`EleutherAI/pythia-70m`
- 单 token 条件：`isolated_token`，序列长度为 1，不使用 LM head、softmax 或 token sampling。
- checkpoint：`step0`、`step1000`、`step16000`、`step143000`。
- 每条轨迹 768 步；图中只展示预注册的 evaluation window `step 512–768`。
- 投影向量在 token 和 checkpoint 之间固定，因此横向比较使用同一坐标系。
- 词频来自 WikiText-2 train split 的重新审计，共 32 个 token、8 档、每档 4 个。
- loss 使用长度 64、512 个固定样本、token-weighted causal cross-entropy。
- test 样本来自 `monology/pile-uncopyrighted` test split。
- train 样本来自 train shard 00 前 {train_metadata['scan_limit']} 个合格记录的确定性 reservoir sample。
- hard natural-language 样本来自本地缓存的 OpenWebMath 两个 Parquet 分片；扫描
  {hard_metadata['raw_records_scanned']} 篇，按预注册文本/数学规则得到
  {hard_metadata['eligible_unique_records']} 篇合格文档，再以 seed
  `{hard_metadata['seed']}` 的 SHA-256 priority 固定抽取 512 篇。
- hard 集要求至少 {hard_metadata['eligibility']['min_chars']} 字符、
  {hard_metadata['eligibility']['min_words']} 个英文词、ASCII 字母占非空白字符比例至少
  {hard_metadata['eligibility']['min_ascii_letter_fraction_nonspace']:.2f}，
  同时要求 OpenWebMath 检测到数学内容且
  `math_score ≥ {hard_metadata['eligibility']['min_math_score']:.2f}`。筛选完全不使用 Pythia loss。
- 三个语料统一使用长度 64、512 个固定样本和 token-weighted causal cross-entropy。

### Token 与词频标注

| 频率档 | 范围 | token（WikiText-2 count） |
|---:|---|---|
{chr(10).join(tokens_by_bin)}

## 投影结果

![step0 的 8 档单 token 投影轨迹](figures/single_token_frequency_projection_step0.png)

![step1000 的 8 档单 token 投影轨迹](figures/single_token_frequency_projection_step1000.png)

![step16000 的 8 档单 token 投影轨迹](figures/single_token_frequency_projection_step16000.png)

![step143000 的 8 档单 token 投影轨迹](figures/single_token_frequency_projection_step143000.png)

每个 panel 中圆点为 evaluation window 起点，叉号为终点。图中已经直接标注 token 文本和词频计数，解决旧图只按 token 文件名分散、难以比较词频的问题。

![词频与动力学指标](figures/frequency_dynamics_metrics.png)

Spearman 相关（`rho` 是 `log10(count+1)` 与指标的秩相关；每个 checkpoint 有 32 个 token）：

| checkpoint | n | Lyapunov rho | p-value |
|---|---:|---:|---:|
{format_correlation_rows(correlations, 'lyapunov_mean')}

最强的 Lyapunov–词频秩相关出现在 `{strongest['checkpoint']}`：
`rho={strongest['spearman_rho']:.3f}`、`p={strongest['p_value']:.3g}`。尽管总样本增加到 32，
每档仍只有 4 个 token，这里应视为探索性规律而不是确定性词频定律。

两个值得记录、但尚未跨 checkpoint 复现的现象：

- `step143000` 的 normalized Frobenius 随词频升高而降低：
  `rho={final_frobenius['spearman_rho']:.3f}`、`p={final_frobenius['p_value']:.3g}`。
  该关系在前三个 checkpoint 不显著，因此更像“最终收敛状态下的候选规律”，不能写成普遍词频定律。
- nearby-growth 在 `step0` 为正相关
  (`rho={step0_nearby['spearman_rho']:.3f}`, `p={step0_nearby['p_value']:.3g}`)，
  到 `step1000` 变成负相关
  (`rho={step1000_nearby['spearman_rho']:.3f}`, `p={step1000_nearby['p_value']:.3g}`)；
  两者均不显著，且方向反转，说明 nearby 指标没有稳定的早期词频规律。

整体上 checkpoint 和状态收敛阶段造成的变化大于八个词频档之间的稳定差异：
早期 checkpoint 的轨迹仍有显著运动，最终 checkpoint 多数轨迹收缩到很小区域；
不同 checkpoint 下相关方向并不保证一致。因此现有证据不支持“词频越高就必然更稳定/更混沌”的单调结论。
`step143000` 投影图中的小范围折线主要处于 float32 数值分辨率附近，不应解释为真实高周期吸引子。

## Train/Test Loss

![The Pile train loss](figures/train_loss_by_checkpoint.png)

![The Pile test loss](figures/test_loss_by_checkpoint.png)

![Train/test 对比与 gap](figures/train_test_loss_comparison.png)

共有 {len(loss_rows)} 个 train/test 共同 checkpoint。最后一个 `{final['checkpoint']}`：
train proxy loss=`{final['train_loss']:.4f}`，test loss=`{final['test_loss']:.4f}`，
test−train=`{final['test_minus_train']:.4f}`。最低 test loss 出现在
`{min_test['checkpoint']}`，为 `{min_test['test_loss']:.4f}`（仅指本次固定样本上的观测最小值）。
由于 train proxy 与 test 来自不同分片和不同固定样本，`test−train` 是语料差异，
不是严格的泛化 gap；其负值不表示测试集优于训练集这一通常意义上的结论。

## 本地困难自然语言 Loss

![本地 OpenWebMath 困难自然语言 loss](figures/local_hard_natural_language_loss.png)

![三个语料的 loss 对比](figures/three_corpus_loss_comparison.png)

本地 hard 集与 The Pile 使用完全相同的 checkpoint 和 loss 口径。在
{hard_above_test}/{len(loss_rows)} 个 checkpoint 上，hard loss 高于 The Pile test；
在 {hard_above_train}/{len(loss_rows)} 个 checkpoint 上高于 train proxy。
最后一个 `{final['checkpoint']}` 的 hard loss 为
`{final['hard_natural_language_loss']:.4f}`，相比 test 高
`{final['hard_minus_test']:+.4f}`、相比 train proxy 高
`{final['hard_minus_train']:+.4f}`。hard 集的观测最低 loss 出现在
`{min_hard['checkpoint']}`，为 `{min_hard['hard_natural_language_loss']:.4f}`。

这里的“困难”是操作性定义：该集合来自数学领域、包含自然语言说明并通过独立规则预先筛选；
其难度由 loss 对比事后核验，而不是用 loss 选择样本。

## 结论

1. 旧实验数据本身已有词频字段；本阶段重新从完整 WikiText-2 audit 中抽取 8 档 × 4 token，并完成同坐标系分组重绘。
2. 单 token 动力学随训练 checkpoint 的变化非常明显；32 个 token 中没有跨 checkpoint 稳定复现的简单词频单调律。
3. Frobenius、Lyapunov、轨迹位移是不同指标；不能用投影图的视觉收缩单独替代稳定性判断。
4. The Pile train/test loss 整体随训练降低；末端是否存在反弹应以 CSV 中相邻 checkpoint 的实际差值判断。
5. 本地 OpenWebMath hard 集在绝大多数 checkpoint 上保持更高 loss，说明模型对数学密集自然语言的建模难度高于当前固定 The Pile 样本。

## 限制

- 词频来自 WikiText-2，不是 Pythia 原始训练语料中的真实 token exposure。
- 每档只有 4 个 token，统计功效仍有限；8 档按 eligible-token rank 分位，而不是等宽 count 区间。
- 序列长度 1 时 attention 退化，结论不能直接外推到正常多 token 生成。
- train loss 是 train-split proxy：只采样 uncopyrighted mirror 的一个 shard，而且该镜像不保证与 Pythia 的精确训练混合一致。
- OpenWebMath hard 集只覆盖本地已有的 2/114 个分片，不代表完整 OpenWebMath；它来自 train split，
  也不能排除与 Pythia 预训练混合存在内容重合。
- “困难自然语言”是领域与文本规则定义，不等价于通用推理能力；数学公式、网页格式和领域术语都会共同抬高 loss。
- `step101000/105000/133000/143000` 的 tokenizer 文件在离线缓存中不完整，因此显式复用了 checkpoint-invariant 的 `step100000` tokenizer。
- float32 下部分 nearby-distance 进入 numerical floor；相关统计对这些行作了排除。

## 可复现产物

- `processed/isolated_token_summary.csv`
- `processed/frequency_bin_summary.csv`
- `processed/frequency_metric_correlations.csv`
- `processed/checkpoint_train_test_loss.csv`
- `processed/checkpoint_three_corpus_loss.csv`
- `figures/single_token_frequency_projection_step0.png`
- `figures/single_token_frequency_projection_step1000.png`
- `figures/single_token_frequency_projection_step16000.png`
- `figures/single_token_frequency_projection_step143000.png`
- `figures/frequency_dynamics_metrics.png`
- `figures/train_loss_by_checkpoint.png`
- `figures/test_loss_by_checkpoint.png`
- `figures/train_test_loss_comparison.png`
- `figures/local_hard_natural_language_loss.png`
- `figures/three_corpus_loss_comparison.png`
- `manifests/the_pile_train.jsonl`
- `manifests/the_pile_train.metadata.json`
- `manifests/open_web_math_local_hard.metadata.json`

生成命令记录在 `RUNBOOK.md`。
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--single-root", required=True)
    parser.add_argument("--test-loss-root", required=True)
    parser.add_argument("--train-loss-root", required=True)
    parser.add_argument(
        "--hard-loss-root",
        help="Root containing raw/open_web_math_local_hard; defaults to output-root.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    figures = output_root / "figures"
    processed = output_root / "processed"
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    trajectory_rows, summary_rows = load_frequency_rows(Path(args.single_root))
    descriptions = bin_descriptions(summary_rows)
    projection_figures(
        trajectory_rows,
        summary_rows,
        descriptions,
        figures,
    )
    metric_figure(summary_rows, figures / "frequency_dynamics_metrics.png")
    correlations, bin_rows = frequency_tables(summary_rows, processed)
    loss_rows = build_loss_table(
        Path(args.train_loss_root), Path(args.test_loss_root), processed, figures
    )
    hard_root = Path(args.hard_loss_root) if args.hard_loss_root else output_root
    loss_rows = extend_with_hard_loss(loss_rows, hard_root, processed, figures)
    train_metadata = json.loads(
        (Path(args.train_loss_root) / "manifests" / "the_pile_train.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    hard_metadata = json.loads(
        (hard_root / "manifests" / "open_web_math_local_hard.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    write_readme(
        output_root,
        descriptions,
        correlations,
        bin_rows,
        loss_rows,
        train_metadata,
        hard_metadata,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "trajectory_rows": len(trajectory_rows),
                "summary_rows": len(summary_rows),
                "loss_checkpoints": len(loss_rows),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
