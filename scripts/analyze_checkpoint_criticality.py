#!/usr/bin/env python
import json
from pathlib import Path
import pandas as pd

root = Path('/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality')
raw = root / 'raw'

def read(pattern):
    rows = []
    for path in raw.glob(pattern):
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    return pd.DataFrame(rows)

dyn = read('*__dynamical_edge.jsonl')
ppl = read('*__perplexity.jsonl')
ppl = ppl[ppl['batch_index'].astype(str) == 'mean'].copy()
dyn['abs_lyapunov'] = dyn['maximal_lyapunov_mean'].abs()
dyn['frob_edge_distance'] = dyn['edge_distance_log']
dyn['final_relative_step_delta'] = dyn['relative_step_deltas'].map(lambda xs: xs[-1] if xs else float('nan'))

dsummary = dyn.groupby('checkpoint', as_index=False).agg(
    dynamics_samples=('sample_index', 'count'),
    frobenius_mean=('normalized_frobenius_geomean', 'mean'),
    frobenius_std=('normalized_frobenius_geomean', 'std'),
    maximal_lyapunov_mean=('maximal_lyapunov_mean', 'mean'),
    maximal_lyapunov_std=('maximal_lyapunov_mean', 'std'),
    abs_lyapunov_mean=('abs_lyapunov', 'mean'),
    final_to_initial_mean=('final_to_initial_separation', 'mean'),
    final_relative_step_delta_mean=('final_relative_step_delta', 'mean'),
    diverged_fraction=('diverged', 'mean'),
    collapsed_fraction=('collapsed', 'mean'),
)
perf = ppl[['checkpoint', 'token_weighted_loss', 'token_weighted_perplexity', 'predicted_token_count']].copy()
merged = dsummary.merge(perf, on='checkpoint', how='inner')
merged['training_step'] = merged['checkpoint'].str.replace('step', '', regex=False).astype(int)
merged = merged.sort_values('training_step')

metrics = ['abs_lyapunov_mean', 'frob_edge_distance'] if 'frob_edge_distance' in merged else ['abs_lyapunov_mean']
merged['frob_edge_distance'] = (-merged['frobenius_mean'].clip(lower=1e-12).map(__import__('math').log)).abs()
corrows = []
for metric in ['abs_lyapunov_mean', 'frob_edge_distance', 'frobenius_mean', 'maximal_lyapunov_mean']:
    corrows.append({
        'metric': metric,
        'pearson_with_token_weighted_loss': merged[metric].corr(merged['token_weighted_loss'], method='pearson'),
        'spearman_with_token_weighted_loss': merged[metric].corr(merged['token_weighted_loss'], method='spearman'),
        'pearson_with_training_step': merged[metric].corr(merged['training_step'], method='pearson'),
        'spearman_with_training_step': merged[metric].corr(merged['training_step'], method='spearman'),
    })
cor = pd.DataFrame(corrows)
phase = pd.crosstab(dyn['checkpoint'], dyn['phase_label']).reset_index()
out = root / 'processed'; out.mkdir(exist_ok=True)
merged.to_csv(out / 'checkpoint_criticality_performance.csv', index=False)
cor.to_csv(out / 'checkpoint_criticality_correlations.csv', index=False)
phase.to_csv(out / 'checkpoint_phase_counts.csv', index=False)
print('MERGED'); print(merged.to_string(index=False))
print('CORRELATIONS'); print(cor.to_string(index=False))
print('PHASE'); print(phase.to_string(index=False))
