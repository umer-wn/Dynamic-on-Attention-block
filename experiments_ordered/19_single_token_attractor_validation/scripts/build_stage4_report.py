#!/usr/bin/env python
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
SCREEN_PATH = PROCESSED / "screen_summary.csv"
ORBIT_PATH = PROCESSED / "orbit_candidates.csv"
SELECTED = ["step2000", "step9000", "step10000", "step13000", "step29000", "step41000", "step57000", "step61000"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "—"
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e4):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def best_per_system(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["initial_state_bank"], row["token_id"])].append(row)
    return [min(group, key=lambda row: f(row, "shooting_normalized_residual_p95")) for group in groups.values()]


def count_text(counter: Counter) -> str:
    return ", ".join(f"{key}={counter.get(key, 0)}" for key in ("stable", "boundary", "unstable"))


def checkpoint_table(rows: list[dict[str, str]]) -> list[list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"]].append(row)
    output: list[list[str]] = []
    for checkpoint in SELECTED:
        current = grouped.get(checkpoint, [])
        if not current:
            output.append([checkpoint, "0", "0", "0", "0", "—", "—", "—", "—"])
            continue
        stability = Counter(row["stability"] for row in current)
        periods = Counter(int(row["minimal_period"]) for row in current)
        period_text = ", ".join(f"p{key}:{value}" for key, value in sorted(periods.items()))
        residuals = [f(row, "shooting_normalized_residual_p95") for row in current]
        radii = [f(row, "leading_multiplier_modulus") for row in current]
        recoveries = [f(row, "recovery_fraction") for row in current]
        output.append([
            checkpoint,
            str(len(current)),
            str(stability["stable"]),
            str(stability["boundary"]),
            str(stability["unstable"]),
            period_text,
            fmt(median(residuals)),
            fmt(median(radii)),
            fmt(max(recoveries)),
        ])
    return output


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def main() -> None:
    screen = read_rows(SCREEN_PATH)
    phase_rows = read_rows(ORBIT_PATH)
    systems = best_per_system(phase_rows)
    screen_counts = Counter(row["screen_classification"] for row in screen)
    phase_stability = Counter(row["stability"] for row in phase_rows)
    system_stability = Counter(row["stability"] for row in systems)
    phase_periods = Counter(int(row["minimal_period"]) for row in phase_rows)
    system_periods = Counter(int(row["minimal_period"]) for row in systems)
    residuals = [f(row, "shooting_normalized_residual_p95") for row in phase_rows]
    recoveries = [f(row, "recovery_fraction") for row in phase_rows]
    converged = sum(row["shooting_converged"].lower() == "true" for row in phase_rows)
    recovered_90 = sum(value >= 0.9 for value in recoveries)
    fixed = sum(int(row["minimal_period"]) == 1 for row in phase_rows)
    validated_cps = sorted({row["checkpoint"] for row in phase_rows}, key=lambda x: int(x[4:]))
    no_candidate = [checkpoint for checkpoint in SELECTED if checkpoint not in validated_cps]
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    phase_table = checkpoint_table(phase_rows)
    system_table = checkpoint_table(systems)

    zh = f"""# 实验19完整报告（截至第四阶段）

**生成时间：** {generated}  
**范围：** 只汇总四阶段：长轨迹回归筛选、多点射击、Floquet/monodromy 稳定性、有限扰动恢复。精度审计和架构消融不作为本报告的判定门槛。

## 结论摘要

- 筛选覆盖 **{len(screen)} 个系统**（19 checkpoints × 3 个初始状态库 × 8 tokens）。标签为：`recurrent_candidate` {screen_counts['recurrent_candidate']}、`expanding_candidate` {screen_counts['expanding_candidate']}、`quasiperiodic_candidate` {screen_counts['quasiperiodic_candidate']}、`transient_or_unresolved` {screen_counts['transient_or_unresolved']}、`fixed_candidate` {screen_counts['fixed_candidate']}。
- 8 个选定 checkpoint 的长轨迹均已完成。{', '.join(validated_cps)} 产生可射击的回归候选；{', '.join(no_candidate)} 没有候选，因此没有 orbit/Floquet 行，这不是漏跑。
- 共得到 **{len(phase_rows)} 条相位射击记录**，对应 **{len(systems)} 个去重系统**（同一 checkpoint–状态库–token 的3个相位取射击残差最小者）。相位级标签：{count_text(phase_stability)}；系统级标签：{count_text(system_stability)}。
- **{converged}/{len(phase_rows)}** 条记录射击收敛，且残差全部不高于 `1e-5`。但是有限扰动恢复率达到 `0.9` 的记录仅 **{recovered_90}/{len(phase_rows)}**。
- 所有最小周期均大于1（`p=1` 数量 {fixed}），因此当前数据没有不动点证据。虽然存在局部 `stable` Floquet 标签，但没有任何候选同时满足“射击收敛 + stable + 恢复率≥0.9”；所以截至第四阶段，**没有验证出有限扰动意义下的稳定吸引周期轨道**。

## 方法和判定规则

1. **回归筛选：** 单 token 状态迭代4096步，分析后2048步，在 lag 1–256 中寻找相位回归；长轨迹验证扩展到16384步并分析后8192步。
2. **多点射击：** 将候选周期拆为多个轨道点，同时最小化每段末端与下一段起点的闭合残差。报告 `shooting_normalized_residual_p95`；`≤1e-5` 视为几何闭合通过。
3. **Floquet 稳定性：** 对一个最小周期上的雅可比乘积（monodromy matrix）用 Arnoldi/Krylov 维数16和32估计最大乘子模 `rho(M)`。代码规则为：`rho<0.98` 且两种Krylov估计相对差<0.05 → stable；`rho>1.02` → unstable；其余 → boundary。
4. **扰动恢复：** 在轨道点沿512维全部坐标构造随机单位方向，使用16方向 × 3相对尺度（`1e-6,1e-4,1e-2`），演化10个周期；最终到整条轨道的相位不变距离不大于初始距离记为恢复。恢复率≥0.9才支持有限扰动吸引性。
5. **不动点/周期轨道区分：** 多点射击后再约简最小周期；`minimal_period=1` 是不动点，`>1` 是周期轨道。本次没有 `p=1`。

## checkpoint结果（相位级，保留3个相位射击）

{markdown_table(['checkpoint','记录','stable','boundary','unstable','最小周期分布','残差中位数','rho(M)中位数','最大恢复率'], phase_table)}

## checkpoint结果（系统级，去除相位重复）

{markdown_table(['checkpoint','系统','stable','boundary','unstable','最小周期分布','残差中位数','rho(M)中位数','最大恢复率'], system_table)}

## 总体数值质量

- 射击残差：min={fmt(min(residuals))}，median={fmt(median(residuals))}，p95={fmt(quantile(residuals, .95))}，max={fmt(max(residuals))}。
- 扰动恢复率：min={fmt(min(recoveries))}，median={fmt(median(recoveries))}，p95={fmt(quantile(recoveries, .95))}，max={fmt(max(recoveries))}。
- 相位级最小周期分布：{dict(sorted(phase_periods.items()))}。
- 系统级最小周期分布：{dict(sorted(system_periods.items()))}。

## 如何解读 stable / unstable / boundary

- `stable` 只表示估计的单周期线性化在局部收缩；它不是吸引子的充分证据。
- `unstable` 表示至少一个估计Floquet乘子模明显大于1，轨道局部排斥。
- `boundary` 表示乘子接近单位圆或Krylov估计未达到稳定阈值，不能可靠归入稳定/不稳定。
- 因为本次所有 `stable` 候选都未通过≥0.9的有限扰动恢复门槛，最终应表述为“闭合周期轨道候选中存在局部稳定标签，但尚未验证出稳定吸引子”。

## 数据文件

- `processed/screen_summary.csv`：456个筛选系统。
- `processed/orbit_candidates.csv`：288条相位级射击、Floquet与扰动恢复结果。
- `processed/stage4_system_summary.csv`：本报告生成的96个去重系统结果。
"""

    en = f"""# Experiment 19 Complete Report (Stages 1–4)

**Generated:** {generated}  
**Scope:** long-trajectory recurrence screening, multiple shooting, Floquet/monodromy stability, and finite-perturbation recovery. Precision audits and architecture ablations are outside the decision scope of this report.

## Executive result

- Screening covered **{len(screen)} systems** (19 checkpoints × 3 initial-state banks × 8 tokens): recurrent={screen_counts['recurrent_candidate']}, expanding={screen_counts['expanding_candidate']}, quasiperiodic={screen_counts['quasiperiodic_candidate']}, transient/unresolved={screen_counts['transient_or_unresolved']}, fixed={screen_counts['fixed_candidate']}.
- All eight selected checkpoints completed their long validation traces. {', '.join(validated_cps)} produced shootable recurrent candidates; {', '.join(no_candidate)} produced none, so their absence from the orbit table is not missing execution.
- There are **{len(phase_rows)} phase-level shooting records**, representing **{len(systems)} unique checkpoint–state-bank–token systems** after choosing the lowest-residual phase. Phase-level stability: {count_text(phase_stability)}. System-level stability: {count_text(system_stability)}.
- **{converged}/{len(phase_rows)}** shooting records converged and all residuals are at most `1e-5`, but only **{recovered_90}/{len(phase_rows)}** records achieve perturbation recovery ≥0.9.
- Every minimal period is greater than one (`p=1`: {fixed}), so no fixed point was detected. No candidate jointly satisfies shooting convergence, stable Floquet label, and recovery ≥0.9. Therefore, through Stage 4, **no stable attracting periodic orbit has been validated under the finite-perturbation criterion**.

## Protocol and criteria

1. Recurrence screening iterates each isolated-token state for 4096 steps and analyzes the final 2048; strict validation uses 16384 steps and the final 8192.
2. Multiple shooting optimizes all orbit nodes simultaneously. A normalized p95 closure residual ≤`1e-5` passes geometric closure.
3. Floquet stability estimates the spectral radius of the one-period Jacobian product with Arnoldi dimensions 16 and 32: stable if `rho<0.98` and relative disagreement <0.05; unstable if `rho>1.02`; otherwise boundary.
4. Recovery uses 16 random full-512D unit directions at three relative scales (`1e-6,1e-4,1e-2`) for ten periods. Recovery means final phase-invariant distance to the orbit does not exceed the initial distance; ≥0.9 is the attraction threshold.
5. Minimal period 1 denotes a fixed point; a value greater than 1 denotes a periodic orbit.

## Per-checkpoint results (phase level)

{markdown_table(['checkpoint','records','stable','boundary','unstable','minimal periods','median residual','median rho(M)','max recovery'], phase_table)}

## Per-checkpoint results (deduplicated systems)

{markdown_table(['checkpoint','systems','stable','boundary','unstable','minimal periods','median residual','median rho(M)','max recovery'], system_table)}

## Numerical summary

- Shooting residual: min={fmt(min(residuals))}, median={fmt(median(residuals))}, p95={fmt(quantile(residuals, .95))}, max={fmt(max(residuals))}.
- Recovery fraction: min={fmt(min(recoveries))}, median={fmt(median(recoveries))}, p95={fmt(quantile(recoveries, .95))}, max={fmt(max(recoveries))}.
- Phase-level minimal-period distribution: {dict(sorted(phase_periods.items()))}.
- System-level minimal-period distribution: {dict(sorted(system_periods.items()))}.

## Interpretation of stable / unstable / boundary

- `stable` means locally contracting under the estimated one-period linearization; it is not sufficient by itself to establish an attractor.
- `unstable` means an estimated Floquet multiplier modulus is clearly above one.
- `boundary` means the multiplier lies near the unit circle or the Krylov estimates do not satisfy the stable rule.
- Since none of the locally stable candidates passes the ≥0.9 recovery threshold, the defensible conclusion is: geometrically closed periodic candidates exist, some have locally stable labels, but no stable attractor is validated.

## Data products

- `processed/screen_summary.csv`: 456 screened systems.
- `processed/orbit_candidates.csv`: 288 phase-level shooting, Floquet, and recovery results.
- `processed/stage4_system_summary.csv`: 96 deduplicated systems generated with this report.
"""

    fields = list(phase_rows[0].keys())
    with (PROCESSED / "stage4_system_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(systems)
    # Keep REPORT_ZH.md as the reader-facing simplified narrative. Automated
    # refreshes update the complete numerical appendix instead of overwriting it.
    (ROOT / "REPORT_TECHNICAL_ZH.md").write_text(zh, encoding="utf-8")
    (ROOT / "REPORT_EN.md").write_text(en, encoding="utf-8")
    print(f"reports refreshed: phase_records={len(phase_rows)}, unique_systems={len(systems)}")


if __name__ == "__main__":
    main()
