import math

import numpy as np
import pandas as pd

from scripts.analyze_pythia_early_single_token_scan import THRESHOLD, paired_bootstrap_delta


def test_coarse_checkpoint_grid_is_first_100_every_four():
    steps = list(range(1000, 98000, 4000))
    assert len(steps) == 25
    assert steps[0] == 1000
    assert steps[-1] == 97000
    assert all(step % 1000 == 0 for step in steps)


def _samples(loss_per_token: float) -> pd.DataFrame:
    counts = np.arange(2, 130)
    return pd.DataFrame(
        {
            "sample_index": np.arange(128),
            "predicted_token_count": counts,
            "nll_sum": counts * loss_per_token,
        }
    )


def test_paired_bootstrap_detects_practical_decrease_and_increase():
    baseline = _samples(3.0)
    decrease = paired_bootstrap_delta(baseline, _samples(3.0 - 2 * THRESHOLD), 1000, 1234)
    increase = paired_bootstrap_delta(baseline, _samples(3.0 + 2 * THRESHOLD), 1000, 1234)
    assert decrease["label"] == "significant_decrease"
    assert increase["label"] == "significant_increase"


def test_practically_tiny_change_is_not_significant_label():
    baseline = _samples(3.0)
    result = paired_bootstrap_delta(baseline, _samples(3.0 + 0.5 * THRESHOLD), 1000, 1234)
    assert result["label"] == "descriptive_increase"
    assert math.isclose(result["delta_loss"], 0.5 * THRESHOLD, rel_tol=1e-10)
