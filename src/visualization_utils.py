from __future__ import annotations

from statistics import median
from typing import Any, Iterable


def projected_poincare_crossings(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return linearly interpolated upward crossings of the median z0 section."""
    ordered = sorted(rows, key=lambda row: int(row["step_index"]))
    if len(ordered) < 2:
        return []
    section = float(median(float(row["projection_0"]) for row in ordered))
    points: list[dict[str, Any]] = []
    previous = ordered[0]
    for current in ordered[1:]:
        previous_value = float(previous["projection_0"]) - section
        current_value = float(current["projection_0"]) - section
        if previous_value <= 0.0 < current_value:
            denominator = current_value - previous_value
            alpha = 0.0 if denominator == 0.0 else -previous_value / denominator
            alpha = min(1.0, max(0.0, alpha))
            previous_z1 = float(previous["projection_1"])
            previous_z2 = float(previous["projection_2"])
            current_z1 = float(current["projection_1"])
            current_z2 = float(current["projection_2"])
            points.append(
                {
                    **current,
                    "section_value": section,
                    "crossing_order": len(points),
                    "crossing_alpha": alpha,
                    "crossing_step": float(previous["step_index"]) + alpha * (
                        float(current["step_index"]) - float(previous["step_index"])
                    ),
                    "poincare_z1": previous_z1 + alpha * (current_z1 - previous_z1),
                    "poincare_z2": previous_z2 + alpha * (current_z2 - previous_z2),
                }
            )
        previous = current
    return points
