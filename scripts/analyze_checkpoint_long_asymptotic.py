#!/usr/bin/env python
import json
import math
from pathlib import Path
import pandas as pd

parent = Path('/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality')
root = parent / 'long_asymptotic'

def read(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.open() if line.strip())
    return pd.DataFrame(rows)

long = read(root.glob('raw/*__dynamical_edge.jsonl'))
short = read(parent.glob('raw/*__dynamical_edge.jsonl'))
ppl = read(parent.glob('raw/*__perplexity.jsonl'))
lag = read(root.glob('raw/*__state_distance_metrics.jsonl'))
traj = read(root.glob('raw/*__dynamics_trajectory.jsonl'))
ppl = ppl[ppl['batch_index'].astype(str) == 'mean'].copy()

def prepare(df, protocol):
    df = df.copy()
    df['protocol'] = protocol
    df['final_relative_step_delta'] = df['relative_step_deltas'].map(lambda xs: xs[-1] if xs else float('nan'))
    df['tail_relative_step_delta'] = df['relative_step_deltas'].map(lambda xs: sum(xs[-5:]) / len(xs[-5:]) if xs else float('nan'))
    df['asymptotic_gate'] = df['tail_relative_step_delta'] < 1e-6
    return df

long = prepare(long, 'burn512_eval128')
long['positive_lyapunov'] = long['maximal_lyapunov_mean'] > 0
short = prepare(short, 'burn64_eval128')
short['positive_lyapunov'] = short['maximal_lyapunov_mean'] > 0
all_rows = pd.concat([short, long], ignore_index=True)
comparison = all_rows.groupby(['protocol', 'checkpoint'], as_index=False).agg(
    samples=('sample_index', 'count'),
    frobenius_mean=('normalized_frobenius_geomean', 'mean'),
    maximal_lyapunov_mean=('maximal_lyapunov_mean', 'mean'),
    maximal_lyapunov_std=('maximal_lyapunov_mean', 'std'),
    final_to_initial_mean=('final_to_initial_separation', 'mean'),
    tail_relative_step_delta_mean=('tail_relative_step_delta', 'mean'),
    asymptotic_gate_fraction=('asymptotic_gate', 'mean'),
    positive_lyapunov_fraction=('positive_lyapunov', 'mean'),
)

valid = long.groupby('checkpoint', as_index=False).agg(
    dynamics_samples=('sample_index', 'count'),
    frobenius_mean=('normalized_frobenius_geomean', 'mean'),
    frobenius_std=('normalized_frobenius_geomean', 'std'),
    maximal_lyapunov_mean=('maximal_lyapunov_mean', 'mean'),
    maximal_lyapunov_std=('maximal_lyapunov_mean', 'std'),
    final_to_initial_mean=('final_to_initial_separation', 'mean'),
    tail_relative_step_delta_mean=('tail_relative_step_delta', 'mean'),
    asymptotic_gate_fraction=('asymptotic_gate', 'mean'),
    positive_lyapunov_fraction=('positive_lyapunov', 'mean'),
)
perf = ppl[['checkpoint', 'token_weighted_loss', 'token_weighted_perplexity', 'predicted_token_count']]
valid = valid.merge(perf, on='checkpoint', how='inner')
valid['training_step'] = valid['checkpoint'].str.replace('step', '', regex=False).astype(int)
valid['abs_lyapunov'] = valid['maximal_lyapunov_mean'].abs()
valid['frob_edge_distance'] = valid['frobenius_mean'].map(lambda x: abs(math.log(max(x, 1e-12))))
valid = valid.sort_values('training_step')

fully_valid = valid[valid['asymptotic_gate_fraction'] == 1.0]
corrows = []
for metric in ['abs_lyapunov', 'frob_edge_distance', 'maximal_lyapunov_mean', 'frobenius_mean']:
    corrows.append({
        'metric': metric,
        'valid_checkpoint_count': len(fully_valid),
        'pearson_with_loss': fully_valid[metric].corr(fully_valid['token_weighted_loss'], method='pearson') if len(fully_valid) > 1 else float('nan'),
        'spearman_with_loss': fully_valid[metric].corr(fully_valid['token_weighted_loss'], method='spearman') if len(fully_valid) > 1 else float('nan'),
    })
cor = pd.DataFrame(corrows)
phase = pd.crosstab([long['checkpoint'], long['asymptotic_gate']], long['phase_label']).reset_index()
norms = traj.groupby('checkpoint', as_index=False).agg(mean_state_norm=('state_norm', 'mean'))
lag_summary = lag.groupby(['checkpoint', 'lag_window'], as_index=False).agg(
    lag_distance_mean=('lag_distance_mean', 'mean'),
    lag_distance_min_mean=('lag_distance_min', 'mean'),
)
lag_summary = lag_summary.merge(norms, on='checkpoint', how='left')
lag_summary['normalized_lag_distance_mean'] = lag_summary['lag_distance_mean'] / lag_summary['mean_state_norm'].clip(lower=1e-12)
lag_summary['normalized_lag_distance_min_mean'] = lag_summary['lag_distance_min_mean'] / lag_summary['mean_state_norm'].clip(lower=1e-12)

stationarity_rows = []
for (checkpoint, sample_index), group in traj.groupby(['checkpoint', 'sample_index']):
    group = group.sort_values('step_index')
    split = len(group) // 2
    first, second = group.iloc[:split], group.iloc[split:]
    n1, n2 = first['state_norm'].mean(), second['state_norm'].mean()
    d1, d2 = first['relative_step_delta'].mean(), second['relative_step_delta'].mean()
    pstd = max(float(group['projection_value'].std()), 1e-12)
    stationarity_rows.append({
        'checkpoint': checkpoint,
        'sample_index': sample_index,
        'state_norm_half_relative_drift': abs(n2 - n1) / max(abs(n1), 1e-12),
        'relative_step_half_relative_drift': abs(d2 - d1) / max(abs(d1), 1e-12),
        'projection_half_mean_shift_in_std': abs(second['projection_value'].mean() - first['projection_value'].mean()) / pstd,
    })
stationarity = pd.DataFrame(stationarity_rows)
stationarity_summary = stationarity.groupby('checkpoint', as_index=False).agg(
    state_norm_drift_mean=('state_norm_half_relative_drift', 'mean'),
    state_norm_drift_max=('state_norm_half_relative_drift', 'max'),
    relative_step_drift_mean=('relative_step_half_relative_drift', 'mean'),
    projection_shift_std_mean=('projection_half_mean_shift_in_std', 'mean'),
)

out = root / 'processed'; out.mkdir(exist_ok=True)
comparison.to_csv(out / 'short_vs_long_comparison.csv', index=False)
valid.to_csv(out / 'long_checkpoint_criticality_performance.csv', index=False)
cor.to_csv(out / 'asymptotic_valid_correlations.csv', index=False)
phase.to_csv(out / 'long_phase_and_gate_counts.csv', index=False)
lag_summary.to_csv(out / 'long_recurrence_lag_summary.csv', index=False)
stationarity.to_csv(out / 'long_stationarity_by_sample.csv', index=False)
stationarity_summary.to_csv(out / 'long_stationarity_summary.csv', index=False)
print('LONG'); print(valid.to_string(index=False))
print('SHORT_VS_LONG'); print(comparison.to_string(index=False))
print('VALID_CORRELATIONS'); print(cor.to_string(index=False))
print('PHASE'); print(phase.to_string(index=False))
print('RECURRENCE'); print(lag_summary.to_string(index=False))
print('STATIONARITY'); print(stationarity_summary.to_string(index=False))
