#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment_plotting import save_figure


REPRESENTATIVE_TOKENS = [35408, 34456, 10692, 18564]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan")
    parser.add_argument("--report", default="reports/pythia_early_training_frobenius_scan_report.md")
    args = parser.parse_args()
    root = Path(args.root)
    processed = root / "processed"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    loss = pd.read_csv(processed / "checkpoint_loss.csv").sort_values("training_step")
    deltas = pd.read_csv(processed / "adjacent_loss_deltas.csv")
    selection_path = root / "status" / "adaptive_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    stage_style = {"coarse": ("o", "#3976b9"), "adaptive": ("D", "#e68a2e"), "sentinel": ("s", "#777777")}
    for stage, group in loss.groupby("sampling_stage"):
        marker, color = stage_style.get(stage, ("o", "#3976b9"))
        axes[0].plot(group.training_step, group.token_weighted_loss, marker=marker, color=color, linestyle="none", label=stage)
        axes[1].plot(group.training_step, group.token_weighted_perplexity, marker=marker, color=color, linestyle="none", label=stage)
    axes[0].plot(loss.training_step, loss.token_weighted_loss, color="#bbbbbb", linewidth=1, zorder=0)
    axes[1].plot(loss.training_step, loss.token_weighted_perplexity, color="#bbbbbb", linewidth=1, zorder=0)
    axes[0].set(xlabel="training step", ylabel="token-weighted test loss", title="Fixed-128-sample test loss")
    axes[1].set(xlabel="training step", ylabel="perplexity", title="Fixed-128-sample test perplexity")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "checkpoint_test_loss.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = deltas.label.map(
        {"significant_decrease": "#3976b9", "significant_increase": "#e68a2e",
         "descriptive_decrease": "#8bb6dc", "descriptive_increase": "#efbd88", "no_change": "#888888"}
    ).fillna("#888888")
    centers = (deltas.step_a + deltas.step_b) / 2
    for index, row in deltas.reset_index(drop=True).iterrows():
        center = (row.step_a + row.step_b) / 2
        ax.errorbar(
            [center], [row.delta_loss],
            yerr=[[row.delta_loss - row.ci95_low], [row.ci95_high - row.delta_loss]],
            fmt="none", ecolor=colors.iloc[index], alpha=0.75, capsize=2,
        )
    ax.scatter(centers, deltas.delta_loss, c=colors, s=28)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(math.log(1.01), color="#e68a2e", linestyle="--", linewidth=0.8)
    ax.axhline(-math.log(1.01), color="#3976b9", linestyle="--", linewidth=0.8)
    ax.set(xlabel="midpoint of adjacent evaluated checkpoints", ylabel="Delta test loss (right - left)",
           title="Paired-bootstrap adjacent loss changes")
    fig.tight_layout()
    save_figure(fig, figures / "adjacent_loss_bootstrap.png")

    frob_path = processed / "token_frobenius.csv"
    checkpoint_frob_path = processed / "checkpoint_frobenius.csv"
    if frob_path.exists() and checkpoint_frob_path.exists():
        frob = pd.read_csv(frob_path).sort_values(["token_id", "training_step"])
        checkpoint_frob = pd.read_csv(checkpoint_frob_path).sort_values("training_step")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for _, group in frob.groupby("token_id"):
            ax.plot(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.35, linewidth=0.8)
            ax.scatter(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.45, s=9)
        ax.fill_between(checkpoint_frob.training_step, checkpoint_frob.q25, checkpoint_frob.q75,
                        color="#3976b9", alpha=0.18, label="token IQR")
        ax.plot(checkpoint_frob.training_step, checkpoint_frob["median"], color="#3976b9", marker="o",
                linewidth=2, label="token median")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.9, label="identity reference")
        ax.set(xlabel="training step", ylabel="normalized Frobenius ||J||F/sqrt(512)",
               title="Exact t=767 single-token Jacobian across training")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, figures / "checkpoint_normalized_frobenius.png")

        padding = max(0.002, 0.15 * (frob.normalized_frobenius.max() - frob.normalized_frobenius.min()))
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for _, group in frob.groupby("token_id"):
            ax.plot(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.4, linewidth=0.8)
            ax.scatter(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.5, s=10)
        ax.fill_between(checkpoint_frob.training_step, checkpoint_frob.q25, checkpoint_frob.q75,
                        color="#3976b9", alpha=0.18)
        ax.plot(checkpoint_frob.training_step, checkpoint_frob["median"], color="#3976b9", marker="o", linewidth=2)
        ax.set_ylim(frob.normalized_frobenius.min() - padding, frob.normalized_frobenius.max() + padding)
        ax.set(xlabel="training step", ylabel="normalized Frobenius ||J||F/sqrt(512)",
               title="Zoomed token dispersion (identity reference is outside this range)")
        fig.tight_layout()
        save_figure(fig, figures / "checkpoint_normalized_frobenius_zoom.png")

        pivot = frob.pivot_table(index="token_id", columns="training_step", values="normalized_frobenius")
        fig, ax = plt.subplots(figsize=(13, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index.astype(str))
        tick_positions = np.linspace(0, len(pivot.columns) - 1, min(12, len(pivot.columns))).round().astype(int)
        ax.set_xticks(tick_positions, labels=[str(pivot.columns[index]) for index in tick_positions], rotation=45, ha="right")
        ax.set(xlabel="training step", ylabel="token id", title="Token x checkpoint normalized Frobenius")
        fig.colorbar(image, ax=ax, label="||J||F/sqrt(512)")
        fig.tight_layout()
        save_figure(fig, figures / "token_checkpoint_frobenius_heatmap.png")

        merged = checkpoint_frob.merge(loss[["training_step", "token_weighted_loss", "sampling_stage"]], on="training_step")
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        scatter = ax.scatter(merged.token_weighted_loss, merged["median"], c=merged.training_step, cmap="plasma", s=48)
        for row in merged.itertuples():
            ax.annotate(str(int(row.training_step)), (row.token_weighted_loss, row.median), fontsize=6, alpha=0.7)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel="token-weighted test loss", ylabel="median normalized Frobenius",
               title="Training ability proxy versus local RMS Jacobian gain")
        fig.colorbar(scatter, ax=ax, label="training step")
        fig.tight_layout()
        save_figure(fig, figures / "loss_vs_normalized_frobenius.png")

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(loss.training_step, loss.token_weighted_loss, color="#e68a2e", marker="o")
        axes[0].set_ylabel("test loss")
        axes[1].fill_between(checkpoint_frob.training_step, checkpoint_frob.q25, checkpoint_frob.q75,
                             color="#3976b9", alpha=0.18)
        axes[1].plot(checkpoint_frob.training_step, checkpoint_frob["median"], color="#3976b9", marker="o")
        axes[1].axhline(1, color="black", linestyle="--", linewidth=0.8)
        axes[1].set(xlabel="training step", ylabel="normalized Frobenius")
        fig.suptitle("Aligned loss and exact single-token Jacobian summaries")
        fig.tight_layout()
        save_figure(fig, figures / "aligned_loss_and_frobenius.png")

        condition_path = processed / "token_frobenius_all_conditions.csv"
        condition_summary_path = processed / "checkpoint_frobenius_conditions.csv"
        if condition_path.exists() and condition_summary_path.exists():
            conditions = pd.read_csv(condition_path)
            condition_summary = pd.read_csv(condition_summary_path)
            names = [name for name in ["self_t0", "tail_t767", "common_step1000_state"] if name in set(conditions.condition)]
            fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), squeeze=False, sharey=False)
            for ax, name in zip(axes.flat, names):
                subset = conditions[conditions.condition == name]
                summary_subset = condition_summary[condition_summary.condition == name]
                for _, group in subset.groupby("token_id"):
                    ax.plot(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.35, linewidth=0.8)
                    ax.scatter(group.training_step, group.normalized_frobenius, color="#9aa0a6", alpha=0.45, s=9)
                ax.fill_between(summary_subset.training_step, summary_subset.q25, summary_subset.q75,
                                color="#3976b9", alpha=0.18)
                ax.plot(summary_subset.training_step, summary_subset["median"], color="#3976b9", marker="o", linewidth=2)
                ax.set(xlabel="training step", ylabel="normalized Frobenius", title=name)
            fig.suptitle("State/weight controls for exact single-token Jacobians (zoomed)")
            fig.tight_layout()
            save_figure(fig, figures / "frobenius_state_weight_controls.png")

        trajectory_frames: list[pd.DataFrame] = []
        for path in sorted((root / "raw").glob("step*/trajectories.jsonl")):
            frame = pd.read_json(path, lines=True)
            frame = frame[frame.token_id.isin(REPRESENTATIVE_TOKENS)]
            if not frame.empty:
                trajectory_frames.append(frame)
        if trajectory_frames:
            trajectory = pd.concat(trajectory_frames, ignore_index=True)
            for token_id in REPRESENTATIVE_TOKENS:
                selected = trajectory[trajectory.token_id == token_id]
                steps = sorted(selected.training_step.unique())
                xlim = selected.projection_0.min(), selected.projection_0.max()
                ylim = selected.projection_1.min(), selected.projection_1.max()
                for chunk_index in range(0, len(steps), 25):
                    chunk = steps[chunk_index: chunk_index + 25]
                    columns = min(5, len(chunk))
                    rows = int(math.ceil(len(chunk) / columns))
                    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
                    for ax, training_step in zip(axes.flat, chunk):
                        group = selected[selected.training_step == training_step].sort_values("dynamics_step")
                        ax.scatter(group.projection_0, group.projection_1, c=group.dynamics_step,
                                   cmap="viridis", s=2, alpha=0.75)
                        ax.scatter(group.projection_0.iloc[0], group.projection_1.iloc[0], marker="x", color="black", s=24)
                        ax.scatter(group.projection_0.iloc[-1], group.projection_1.iloc[-1], marker="o", color="red", s=16)
                        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_title(f"step{int(training_step)}", fontsize=9)
                    for ax in axes.flat[len(chunk):]:
                        ax.axis("off")
                    fig.suptitle(f"Fixed 2D projection trajectory; token {token_id}; x=start, red=end")
                    fig.tight_layout()
                    save_figure(fig, figures / f"trajectory_grid__token{token_id}__part{chunk_index // 25 + 1}.png")

    condition_table = ""
    condition_summary_path = processed / "checkpoint_frobenius_conditions.csv"
    if condition_summary_path.exists():
        table_frame = pd.read_csv(condition_summary_path)[["checkpoint", "condition", "median", "q25", "q75", "std"]]
        lines = ["| checkpoint | condition | median | q25 | q75 | std |", "|---|---|---:|---:|---:|---:|"]
        for row in table_frame.itertuples(index=False):
            lines.append(
                f"| {row.checkpoint} | {row.condition} | {row.median:.8f} | {row.q25:.8f} | {row.q75:.8f} | {row.std:.8f} |"
            )
        condition_table = "\n".join(lines)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    minimum = loss.loc[loss.token_weighted_loss.idxmin()]
    report = f"""# Pythia 早期训练 single-token Frobenius 扫描阶段报告

状态：`{selection.get('search_status', 'analysis_incomplete')}`  
更新时间：由 `plot_pythia_early_single_token_scan.py` 自动生成

## 1. 研究问题

在固定 Pythia-70M 架构、固定 16 个 token 和固定 G1 single-token feedback 算子后，精确 token-level normalized Frobenius 是否随 attention 模型的训练程度和固定测试集 loss 发生可复现变化。

训练 checkpoint 没有在本实验中继续训练。`stepN` 是 Pythia 官方预训练过程中保存的静态权重；对每个静态 checkpoint 分别执行 test loss 和隐藏状态循环。

## 2. 当前完成度

- loss checkpoint 数：{len(loss)}
- 最低观测 loss：{minimum.token_weighted_loss:.6f}（step{int(minimum.training_step)}）
- 自适应搜索状态：`{selection.get('search_status', 'unknown')}`
- 已确认显著反转数：{len(selection.get('confirmed_reversals', []))}
- 完整 dynamics checkpoint 数：{selection.get('full_dynamics_checkpoints', 0)}

本阶段实际运行缓存 fallback checkpoint `step0/step1000/step16000`。原定 `step5000` 从官方 Hub 预取因服务器到 `huggingface.co:443` connection timeout 失败；因此本阶段只作为方法、成本和混杂因素门控，不替代前 100 checkpoint 粗扫描。

## 3. 固定公式与数据流

每个 checkpoint 的 test loss 是固定 WikiText-2 test 前 128 个非空样本上的 token-weighted causal cross-entropy。single-token dynamics 为 `x_(t+1)=F_theta(x_t)`，输入输出均为 `[512]`；模型权重固定，LM head、softmax 和 token sampling 不进入循环。精确 Jacobian shape 为 `[512,512]`，主指标是 `||J||_F/sqrt(512)`。

为避免把训练权重变化与末端吸引子混在一起，指标拆成：当前 checkpoint 自身 embedding 上的 `self_t0`、迭代末端的 `tail_t767`，以及所有 checkpoint 在固定 step1000 token-vector bank 上求导的 `common_step1000_state`。

## 4. 当前数值摘要

{condition_table if condition_table else 'Jacobian control 补测尚未完成。'}

## 5. 图表阅读

- [checkpoint_test_loss.png]({figures / 'checkpoint_test_loss.png'})：蓝色粗扫描、橙色自适应真实 checkpoint、灰色边界 sentinel。连线只帮助阅读，不代表数值插值。
- [adjacent_loss_bootstrap.png]({figures / 'adjacent_loss_bootstrap.png'})：相邻实测 checkpoint 的 loss 差及 paired-bootstrap 95% CI；跨过零线表示统计未决，跨过虚线还需满足 1% PPL 的实际效应阈值。
- [checkpoint_normalized_frobenius.png]({figures / 'checkpoint_normalized_frobenius.png'})：灰色为 16 个 token，蓝线为中位数、带为 IQR；虚线 1 是 identity RMS-gain 参考，不是 Lyapunov 零线。
- [loss_vs_normalized_frobenius.png]({figures / 'loss_vs_normalized_frobenius.png'})：检验 Frobenius 更接近模型能力还是只随 step 漂移。需要联合 token 配对与反转带阅读，不能只凭 checkpoint 均值拟合。
- [frobenius_state_weight_controls.png]({figures / 'frobenius_state_weight_controls.png'})：并列比较 `self_t0/tail_t767/common-state`，用来识别末端吸引子选择的混杂。
- 轨迹 grid：每个 panel 是一个 checkpoint 下单个 token 的 `(z0,z1)` 轨迹；黑叉是初始 embedding，红点是末端。固定轴和固定投影保证可比，但二维重叠不证明高维状态相同。

## 6. 当前结论边界

当前 loss 在两个区间均显著下降，但仅有三个非均匀 checkpoint，不能计算可信的训练相关性或寻找反升段。`tail_t767` 在 step16000 的 token 标准差约 `2.28e-7`，说明不同 token 很可能进入同一末端状态/吸引子；因此仅看 tail Frobenius 会把训练与吸引子选择混合。

在自适应搜索和完整粗扫描完成前，不下“存在真实关系”的结论。最终支持条件包括：大多数 token 配对方向一致、checkpoint 相关 CI 不跨零、leave-one-token-out 稳健，并在 loss 下降后反升的区间出现与 loss 相符的 Frobenius 回转。即使通过，也只是 checkpoint 间关联，不是训练因果；G1 的 seq1 attention 已退化，不能直接推广到完整上下文 attention Jacobian。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figures": str(figures), "status": selection.get("search_status")}))


if __name__ == "__main__":
    main()
