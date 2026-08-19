from __future__ import annotations

import unittest

from src.visualization_utils import projected_poincare_crossings


def row(step: int, z0: float, z1: float = 0.0, z2: float = 0.0):
    return {"step_index": step, "projection_0": z0, "projection_1": z1, "projection_2": z2}


class VisualizationUtilsTest(unittest.TestCase):
    def test_no_crossing(self) -> None:
        self.assertEqual(projected_poincare_crossings([row(0, 1.0), row(1, 1.0)]), [])

    def test_single_crossing(self) -> None:
        points = projected_poincare_crossings([row(0, -1.0), row(1, 1.0, 2.0, 3.0)])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["crossing_alpha"], 0.5)
        self.assertEqual(points[0]["crossing_step"], 0.5)
        self.assertEqual(points[0]["poincare_z1"], 1.0)
        self.assertEqual(points[0]["poincare_z2"], 1.5)

    def test_periodic_crossings(self) -> None:
        rows = [row(i, value) for i, value in enumerate([-1.0, 1.0, -1.0, 1.0, -1.0])]
        self.assertEqual(len(projected_poincare_crossings(rows)), 2)


if __name__ == "__main__":
    unittest.main()
