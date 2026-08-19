import math

import torch

from src.single_token_dynamics import classify_convergence, jacobian_summary, projected_poincare_points


def test_normalized_frobenius_identity_and_contraction():
    hidden = 8
    identity = torch.eye(hidden)
    contraction = 0.25 * identity
    assert abs(jacobian_summary(identity)["normalized_frobenius"] - 1.0) < 1e-7
    assert abs(jacobian_summary(contraction)["normalized_frobenius"] - 0.25) < 1e-7


def test_square_shape_and_checksum_are_stable():
    matrix = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    first = jacobian_summary(matrix)
    second = jacobian_summary(matrix.clone())
    assert first["shape"] == [4, 4]
    assert first["checksum_sha256"] == second["checksum_sha256"]


def test_locked_convergence_labels():
    assert classify_convergence(1e-8, -0.2, [-0.1, -0.2]) == "stable_fixed_point_candidate"
    assert classify_convergence(1e-3, -0.2, [-0.1, -0.2]) == "stable_nonfixed_candidate"
    assert classify_convergence(1e-3, 0.2, [0.1, 0.2]) == "expanding_or_chaotic_candidate"
    assert classify_convergence(1e-8, -0.2, [-0.1, -0.2], True) == "unresolved"
    assert classify_convergence(1e-3, 0.2, [0.1, -0.2]) == "unresolved"


def test_contraction_lyapunov_truth():
    alpha = 0.5
    assert math.isclose(math.log(alpha), -0.6931471805599453)


def _projection_rows(values):
    return [
        {"step": index, "projection_0": z0, "projection_1": index, "projection_2": 2 * index,
         "projection_3": -index}
        for index, z0 in enumerate(values)
    ]


def test_poincare_no_crossing_and_single_crossing():
    assert projected_poincare_points(_projection_rows([0, 0, 0])) == []
    points = projected_poincare_points(_projection_rows([-1, 0, 1]))
    assert len(points) == 1
    assert points[0]["section_z0"] == 0.0
    assert points[0]["z1"] == 1.0


def test_poincare_periodic_upward_crossings():
    points = projected_poincare_points(_projection_rows([-2, 2, -2, 2, 0]))
    assert len(points) == 2
    assert [point["crossing_step"] for point in points] == [0.5, 2.5]
