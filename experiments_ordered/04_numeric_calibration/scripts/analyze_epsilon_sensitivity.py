#!/usr/bin/env python
import json
from pathlib import Path
import pandas as pd

root = Path('/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity')
rows = []
products = []
for label, epsilon in [('eps_1e3', 1e-3), ('eps_1e5', 1e-5), ('eps_1e7', 1e-7)]:
    for path in (root / label / 'raw').glob('*__dynamical_edge.jsonl'):
        for line in path.open():
            row = json.loads(line)
            row['epsilon'] = epsilon
            rows.append(row)
    for path in (root / label / 'raw').glob('*__product_jacobian_metrics.jsonl'):
        for line in path.open():
            row = json.loads(line)
            row['epsilon'] = epsilon
            products.append(row)

df = pd.DataFrame(rows)
pdf = pd.DataFrame(products)
summary = df.groupby('epsilon', as_index=False).agg(
    samples=('sample_index', 'count'),
    frobenius_mean=('normalized_frobenius_geomean', 'mean'),
    frobenius_std=('normalized_frobenius_geomean', 'std'),
    nearby_ratio_mean=('nearby_growth_ratio', 'mean'),
    nearby_log_growth_mean=('nearby_log_growth_per_step', 'mean'),
    nearby_log_growth_std=('nearby_log_growth_per_step', 'std'),
    diverged_fraction=('diverged', 'mean'),
    collapsed_fraction=('collapsed', 'mean'),
)
product = pdf.groupby(['epsilon', 'product_window'], as_index=False).agg(
    product_gain_mean=('product_gain_mean', 'mean'),
    product_log_gain_mean=('product_log_gain_mean', 'mean'),
)
out = root / 'processed'
out.mkdir(exist_ok=True)
summary.to_csv(out / 'epsilon_sensitivity_summary.csv', index=False)
product.to_csv(out / 'epsilon_sensitivity_product_gain.csv', index=False)
print('SUMMARY')
print(summary.to_string(index=False))
print('PRODUCT')
print(product.to_string(index=False))
