#!/usr/bin/env python
import json
from pathlib import Path
import pandas as pd

root = Path('/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement')
rows = []
for path in root.glob('alpha*/raw/*__dynamical_edge.jsonl'):
    rows.extend(json.loads(line) for line in path.open() if line.strip())
df = pd.DataFrame(rows)
df['final_relative_step_delta'] = df['relative_step_deltas'].map(lambda x: x[-1] if x else float('nan'))
summary = df.groupby('residual_alpha', as_index=False).agg(
    samples=('sample_index', 'count'),
    frobenius_mean=('normalized_frobenius_geomean', 'mean'),
    lyapunov_mean=('maximal_lyapunov_mean', 'mean'),
    lyapunov_std_across_samples=('maximal_lyapunov_mean', 'std'),
    lyapunov_probe_std_mean=('maximal_lyapunov_std', 'mean'),
    final_distance_mean=('final_asymptotic_distance', 'mean'),
    final_to_initial_mean=('final_to_initial_separation', 'mean'),
    final_relative_step_delta_mean=('final_relative_step_delta', 'mean'),
    diverged_fraction=('diverged', 'mean'),
    collapsed_fraction=('collapsed', 'mean'),
)
phase = pd.crosstab(df['residual_alpha'], df['phase_label']).reset_index()
out = root / 'processed'; out.mkdir(exist_ok=True)
summary.to_csv(out / 'paper_aligned_lyapunov_summary.csv', index=False)
phase.to_csv(out / 'paper_aligned_phase_counts.csv', index=False)
print(summary.to_string(index=False)); print(phase.to_string(index=False))
