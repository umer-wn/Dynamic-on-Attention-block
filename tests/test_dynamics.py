from __future__ import annotations

import math
import unittest

import torch

from src.dynamics import (
    apply_operator_update,
    estimate_normalized_frobenius,
    lagged_state_distance_metrics,
    make_projection_vector,
    make_projection_bank,
    maximal_lyapunov_metrics,
    multi_step_jacobian_product_metrics,
    nearby_growth_metrics,
    project_state,
    run_feedback_trajectory,
    trajectory_summary_rows,
)


class DynamicsTest(unittest.TestCase):
    def test_output_scale_controls_operator_gain(self) -> None:
        x = torch.randn(1, 8)
        y = apply_operator_update(x, x, mode="direct", output_scale=2.5)

        self.assertTrue(torch.allclose(y, 2.5 * x))

    def test_maximal_lyapunov_matches_linear_maps(self) -> None:
        x0 = torch.randn(1, 8)
        for scale in (0.5, 1.0, 2.0):
            operator = lambda x, s=scale: s * x
            result = run_feedback_trajectory(operator, x0, burn_in_steps=0, eval_steps=8)
            metrics = maximal_lyapunov_metrics(operator, result.eval_states, probes=2)
            self.assertTrue(math.isclose(metrics["maximal_lyapunov_mean"], math.log(scale), abs_tol=1e-6))

    def test_trajectory_records_initial_and_relative_distances(self) -> None:
        x0 = torch.randn(1, 8)
        result = run_feedback_trajectory(lambda x: x, x0, burn_in_steps=0, eval_steps=2)
        rows = trajectory_summary_rows(result)
        self.assertGreater(result.initial_perturbation_distance, 0.0)
        self.assertEqual(rows[0]["relative_step_delta"], 0.0)

    def test_identity_operator_has_unit_normalized_frobenius(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(1, 8)

        result = run_feedback_trajectory(
            lambda x: x,
            x0,
            burn_in_steps=1,
            eval_steps=2,
        )
        frob = estimate_normalized_frobenius(
            lambda x: x,
            result.eval_states,
            probes=4,
            probe_distribution="rademacher",
        )

        self.assertEqual(result.phase_label, "stable_fixed_like")
        self.assertTrue(math.isclose(frob.geometric_mean_normalized_frobenius, 1.0, rel_tol=1e-6))

    def test_contracting_operator_is_below_edge(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(1, 8)
        operator = lambda x: 0.5 * x

        result = run_feedback_trajectory(
            operator,
            x0,
            burn_in_steps=1,
            eval_steps=2,
        )
        frob = estimate_normalized_frobenius(
            operator,
            result.eval_states,
            probes=4,
            probe_distribution="rademacher",
        )

        self.assertLess(frob.geometric_mean_normalized_frobenius, 1.0)

    def test_lagged_distance_and_nearby_growth_are_finite(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(1, 8)
        result = run_feedback_trajectory(
            lambda x: 0.9 * x,
            x0,
            burn_in_steps=1,
            eval_steps=4,
        )

        lag_rows = lagged_state_distance_metrics(result.eval_states, [1, 2])
        nearby = nearby_growth_metrics(result.nearby_distances, len(result.nearby_distances))

        self.assertEqual([row["lag_window"] for row in lag_rows], [1, 2])
        self.assertTrue(math.isfinite(nearby["nearby_growth_ratio"]))
        self.assertLess(nearby["nearby_log_growth_per_step"], 0.0)

    def test_product_jacobian_gain_for_linear_operator(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(1, 8)
        operator = lambda x: 0.5 * x
        result = run_feedback_trajectory(
            operator,
            x0,
            burn_in_steps=1,
            eval_steps=3,
        )

        rows = multi_step_jacobian_product_metrics(
            operator,
            result.eval_states,
            windows=[2],
            probes=2,
            probe_distribution="rademacher",
        )

        self.assertEqual(rows[0]["product_window"], 2)
        self.assertTrue(math.isclose(rows[0]["product_gain_mean"], 0.25, rel_tol=1e-6))

    def test_fixed_random_projection_is_reused_deterministically(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(1, 4, 8)
        v1 = make_projection_vector(x0, None, "fixed_random", seed=123)
        v2 = make_projection_vector(x0, None, "fixed_random", seed=123)
        v3 = make_projection_vector(x0, None, "fixed_random", seed=124)

        self.assertTrue(torch.allclose(v1, v2))
        self.assertFalse(torch.allclose(v1, v3))
        self.assertTrue(math.isfinite(project_state(x0, "fixed_random", v1)))

    def test_projection_bank_is_shared_and_deterministic(self) -> None:
        x0 = torch.randn(1, 4, 8)
        first = make_projection_bank(x0, None, "fixed_random", 3, 123, sample_index=0, shared_across_samples=True)
        second = make_projection_bank(x0, None, "fixed_random", 3, 123, sample_index=7, shared_across_samples=True)
        private = make_projection_bank(x0, None, "fixed_random", 3, 123, sample_index=7, shared_across_samples=False)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.allclose(first[0], private[0]))


if __name__ == "__main__":
    unittest.main()
