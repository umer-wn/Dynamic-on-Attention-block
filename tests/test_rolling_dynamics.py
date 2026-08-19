import math
import unittest
from types import SimpleNamespace

import torch

from src.rolling_dynamics import (
    SoftNextTokenRollingOperator,
    analytic_shift_normalized_frobenius,
    estimate_innovation_frobenius,
    estimate_maximal_lyapunov,
    shift_only_operator,
)


class FakeBackbone(torch.nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeModel(torch.nn.Module):
    def __init__(self, vocab=5, hidden=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.gpt_neox = FakeBackbone()

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


class LinearInnovation:
    def __init__(self, scale):
        self.scale = float(scale)

    def next_embedding(self, x):
        return self.scale * x[:, -1, :]

    def __call__(self, x):
        return torch.cat([x[:, 1:, :], self.next_embedding(x).unsqueeze(1)], dim=1)


class RollingDynamicsTests(unittest.TestCase):
    def test_shift_only_formula_and_jvp(self):
        x = torch.randn(1, 4, 3, requires_grad=True)
        v = torch.ones_like(x)
        _, jv = torch.autograd.functional.jvp(shift_only_operator, (x,), (v,))
        self.assertAlmostEqual(float(jv.pow(2).sum()), 9.0, places=6)
        self.assertAlmostEqual(analytic_shift_normalized_frobenius(4), math.sqrt(3 / 4), places=8)

    def test_soft_operator_appends_expected_embedding(self):
        torch.manual_seed(2)
        model = FakeModel()
        operator = SoftNextTokenRollingOperator(model, temperature=0.7)
        x = torch.randn(1, 4, 3)
        next_embedding, probs = operator.next_embedding_and_probs(x)
        manual_logits = model.head(x[:, -1, :])
        manual_probs = torch.softmax(manual_logits / 0.7, dim=-1)
        manual_embedding = manual_probs @ model.embedding.weight
        actual = operator(x)
        self.assertEqual(tuple(actual.shape), (1, 4, 3))
        self.assertTrue(torch.allclose(probs, manual_probs))
        self.assertTrue(torch.allclose(next_embedding, manual_embedding))
        self.assertTrue(torch.allclose(actual[:, :-1, :], x[:, 1:, :]))
        self.assertTrue(torch.allclose(actual[:, -1, :], manual_embedding))

    def test_innovation_decomposition_matches_linear_truth(self):
        operator = LinearInnovation(scale=2.0)
        states = [torch.randn(1, 4, 3), torch.randn(1, 4, 3)]
        result = estimate_innovation_frobenius(operator, states, probes=4, seed=9)
        self.assertAlmostEqual(result["innovation_geomean"], 1.0, places=6)
        self.assertAlmostEqual(result["innovation_output_geomean"], 2.0, places=6)
        self.assertAlmostEqual(result["total_geomean"], math.sqrt(7 / 4), places=6)

    def test_lyapunov_tracks_dominant_linear_innovation(self):
        operator = LinearInnovation(scale=1.2)
        states = [torch.randn(1, 4, 3) for _ in range(96)]
        result = estimate_maximal_lyapunov(operator, states, probes=1, seed=4)
        self.assertAlmostEqual(result["maximal_lyapunov_mean"], math.log(1.2), delta=0.03)


if __name__ == "__main__":
    unittest.main()
