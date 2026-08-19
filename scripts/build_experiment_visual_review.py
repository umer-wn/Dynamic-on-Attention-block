#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization_utils import projected_poincare_crossings


REPO = Path('/data1/luohaoming/model_feature')
OUT = Path('/home/luohaoming/model_feature_reports/experiment_visualization_review')
CURATED = REPO / 'reports/assets/experiment_visualization'
PHASE_COLORS = {'chaotic': '#e6863b', 'unresolved': '#3a9d5d', 'stable': '#3976b9', 'historical': '#777777'}
CHECKPOINT_ORDER = ['step0', 'step1000', 'step16000', 'step143000']
CHECKPOINT_STEP = {'step0': 0, 'step1000': 1000, 'step16000': 16000, 'step143000': 143000}
CHECKPOINT_STATUS = {'step0': 'chaotic', 'step1000': 'unresolved', 'step16000': 'stable', 'step143000': 'stable'}
PHASE_MARKERS = {'chaotic': '^', 'unresolved': 'D', 'stable': 'o', 'historical': 'x'}
manifest: list[dict[str, object]] = []

PHASE_METADATA = {
    2: ('Pythia-70M early dynamics', 'source trajectory samples', 'historical runs; no reinterpretation as corrected Lyapunov'),
    3: ('Pythia/GPT-2/Qwen where applicable', 'source experiment samples', 'grouped by model/operator/temperature; historical nearby metrics explicitly flagged'),
    4: ('Pythia-70M calibration runs', 'matched calibration samples', 'grouped by output scale or epsilon'),
    5: ('Pythia-70M controlled residual runs', 'matched samples across alpha', 'paper-aligned Benettin protocol; identity and contraction controls'),
    6: ('Pythia-70M step0/step1000/step16000/step143000', '8 matched samples per checkpoint unless the performance summary is plotted', 'seq64; long protocol burn512/eval256; no hidden sample exclusion'),
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_jsonl_glob(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(pattern)):
        rows.extend(json.loads(line) for line in path.open(encoding='utf-8') if line.strip())
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.grid(True, color='#d0d0d0', linewidth=0.6, alpha=0.55)
    ax.spines[['top', 'right']].set_visible(False)


def save_figure(fig, phase: int, name: str, title: str, sources: list[str], question: str, interpretation: str, caveat: str, evidence: str = 'current'):
    phase_dir = OUT / f'phase_{phase:02d}'
    phase_dir.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    expanded_sources = []
    for source in sources:
        matches = sorted(glob.glob(source)) if any(c in source for c in '*?[') else [source]
        expanded_sources.extend(matches or [source])
    axes_units = '; '.join(
        f'x={ax.get_xlabel() or "unlabeled"} ({ax.get_xscale()}), y={ax.get_ylabel() or "unlabeled"} ({ax.get_yscale()})'
        for ax in fig.axes if ax.get_visible()
    )
    checkpoints, sample_count, filtering = PHASE_METADATA[phase]
    path = phase_dir / f'{name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    curated_path = CURATED / f'phase_{phase:02d}__{name}.png'
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        image = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        image.save(curated_path, optimize=True, compress_level=9)
    note = path.with_suffix('.md')
    note.write_text('\n'.join([
        f'# {title}', '', f'- Figure: `{path}`', f'- Curated copy: `{curated_path}`',
        f'- Sources: {"; ".join(expanded_sources)}', f'- Checkpoints/models: {checkpoints}',
        f'- Sample count: {sample_count}', f'- Filtering: {filtering}', f'- Axes and units: {axes_units}',
        f'- Research question: {question}',
        f'- Interpretation: {interpretation}', f'- Caveat: {caveat}', f'- Evidence status: {evidence}', ''
    ]), encoding='utf-8')
    manifest.append({'phase': phase, 'figure': name, 'title': title, 'full_path': str(path), 'curated_path': str(curated_path), 'sources': '; '.join(expanded_sources), 'checkpoints_models': checkpoints, 'sample_count': sample_count, 'filtering': filtering, 'axes_units': axes_units, 'question': question, 'interpretation': interpretation, 'caveat': caveat, 'evidence_status': evidence})


def phase2():
    traj_path=REPO/'results/processed/pythia_dynamical_edge_projection64__trajectory_summary.csv'; traj=read_csv(traj_path)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for _,g in traj.groupby('sample_index'):
        g=g.sort_values('step_index'); axes[0].plot(g.state_norm,g.step_delta,alpha=.7); axes[1].plot(g.step_index,g.nearby_distance,alpha=.75)
    axes[0].set(xlabel='state norm',ylabel='step delta',title='Early phase projection (historical)'); axes[1].set_yscale('log'); axes[1].set(xlabel='eval step',ylabel='nearby distance (log)',title='Nearby trajectories'); [style_axes(a) for a in axes]; fig.tight_layout()
    save_figure(fig,2,'early-dynamics-diagnostics','Early dynamics diagnostics',[str(traj_path)],'What did the first feedback trajectories look like?','They motivated the paper-method refactor and longer asymptotic checks.','The old nearby ratio is not paper method 3; this panel is historical.', 'historical')

    ret_path=REPO/'results/processed/pythia_dynamical_edge_projection64__return_map_points.csv'; ret=read_csv(ret_path)
    fig,ax=plt.subplots(figsize=(5.5,5.5)); sc=ax.scatter(ret.projection_value,ret.projection_next,c=ret.step_index,s=14,alpha=.7,cmap='viridis'); ax.set(xlabel='z_t',ylabel='z_(t+1)',title='Early fixed-projection return map'); fig.colorbar(sc,ax=ax,label='step'); style_axes(ax)
    save_figure(fig,2,'early-return-map','Early fixed-projection return map',[str(ret_path)],'Does the early trajectory show a simple return-map structure?','The projection displays the finite feedback orbit used in early diagnostics.','One projection can hide structure and this run predates corrected Lyapunov.', 'historical')


def phase3():
    summaries=[
      REPO/'experiments/pythia_dynamical_edge_normal/results/processed/pythia_dynamical_edge_normal__dynamical_edge_summary.csv',
      REPO/'experiments/gpt2_dynamical_edge_normal/results/processed/gpt2_dynamical_edge_normal__dynamical_edge_summary.csv',
      REPO/'experiments/qwen_dynamical_edge_normal_long/results/processed/qwen_dynamical_edge_normal_long__dynamical_edge_summary.csv']
    parts=[read_csv(p) for p in summaries]; cross=pd.concat(parts,ignore_index=True)
    raw_cross=[]; raw_cross_sources=[]
    for idx,p in enumerate(summaries):
        raw_dir=p.parents[1]/'raw'; g=read_jsonl_glob(raw_dir,'*dynamical_edge.jsonl'); g['category']=idx; raw_cross.append(g); raw_cross_sources.append(str(raw_dir/'*dynamical_edge.jsonl'))
    raw_cross=pd.concat(raw_cross,ignore_index=True)
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(cross)); colors=['#3976b9','#3a9d5d','#e6863b']; ax.bar(x,cross.mean_normalized_frobenius,color=colors,alpha=.82)
    rng=np.random.default_rng(1234)
    for idx in x:
        g=raw_cross[raw_cross.category==idx]; ax.scatter(idx+rng.normal(0,.035,len(g)),g.normalized_frobenius_mean,color='#222222',s=24,zorder=3)
    ax.axhline(1,color='#555',ls='--'); ax.set_xticks(x,[m.split('/')[-1] for m in cross.model]); ax.set(ylabel='mean normalized Frobenius',title='Cross-model constructed-operator comparison'); style_axes(ax)
    save_figure(fig,3,'cross-model-frobenius','Cross-model Frobenius comparison',[str(p) for p in summaries]+raw_cross_sources,'Does the chosen feedback operator behave similarly across models?','All three average gains are below one, but convergence differs by model.','These historical Frobenius rows do not by themselves establish phase.')

    op_paths=[REPO/'experiments/pythia_dynamical_edge_normal/results/processed/pythia_dynamical_edge_normal__dynamical_edge_summary.csv',REPO/'experiments/pythia_dynamical_edge_norm_matched_probe/results/processed/pythia_dynamical_edge_norm_matched_probe__dynamical_edge_summary.csv',REPO/'experiments/pythia_dynamical_edge_residual_alpha01_probe/results/processed/pythia_dynamical_edge_residual_alpha01_probe__dynamical_edge_summary.csv']
    ops=pd.concat([read_csv(p) for p in op_paths],ignore_index=True); labels=['direct','norm-matched','residual alpha=.1']; raw_ops=[]; raw_op_sources=[]
    for idx,p in enumerate(op_paths):
        raw_dir=p.parents[1]/'raw'; g=read_jsonl_glob(raw_dir,'*dynamical_edge.jsonl'); g['category']=idx; raw_ops.append(g); raw_op_sources.append(str(raw_dir/'*dynamical_edge.jsonl'))
    raw_ops=pd.concat(raw_ops,ignore_index=True)
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(labels)); ax.bar(x,ops.mean_normalized_frobenius,color=['#3976b9','#777777','#3a9d5d'],alpha=.82); rng=np.random.default_rng(1234)
    for idx in x:
        g=raw_ops[raw_ops.category==idx]; ax.scatter(idx+rng.normal(0,.035,len(g)),g.normalized_frobenius_mean,color='#222222',s=24,zorder=3)
    ax.set_xticks(x,labels); ax.axhline(1,color='#555',ls='--'); ax.set(ylabel='mean normalized Frobenius',title='Operator choice changes the measured regime'); style_axes(ax)
    save_figure(fig,3,'operator-choice-comparison','Operator choice comparison',[str(p) for p in op_paths]+raw_op_sources,'How sensitive is criticality to operator construction?','Residual mixing can place average gain near one while norm matching can collapse trajectories.','Artificial operator changes are calibration/intervention, not native model properties.')

    temp_path=REPO/'experiments/pythia_embedding_expectation_temp_sweep/results/temperature_sweep_summary.csv'; temp=read_csv(temp_path); temp['temperature']=temp.label.str.replace('T=','',regex=False).astype(float)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].plot(temp.temperature,temp.mean_frob,marker='o'); axes[0].axhline(1,color='#555',ls='--'); axes[0].set(ylabel='mean Frobenius')
    axes[1].plot(temp.temperature,temp.nearby_log_growth_mean,marker='o'); axes[1].axhline(0,color='#555',ls='--'); axes[1].set(ylabel='historical nearby log growth')
    axes[2].plot(temp.temperature,temp.settled_fraction,marker='o'); axes[2].set(ylabel='settled fraction',ylim=(-.05,1.05))
    for a in axes: a.set_xlabel('temperature'); style_axes(a)
    fig.suptitle('Embedding-expectation temperature sweep'); fig.tight_layout()
    save_figure(fig,3,'temperature-sweep','Embedding-expectation temperature sweep',[str(temp_path)],'Does a different equal-dimensional operator create a controllable transition?','Temperature changes gain, separation, and convergence, demonstrating operator dependence.','Nearby growth is an old diagnostic and is labeled historical.', 'historical')


def phase4():
    gain_path=REPO/'results/processed/pythia_operator_gain_sweep__summary.csv'; gain=read_csv(gain_path)
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.errorbar(gain.output_scale,gain.frobenius_mean,yerr=gain.frobenius_std,marker='o',capsize=4,color='#3976b9'); ax.axhline(1,color='#555',ls='--'); ax.set(xlabel='output scale β',ylabel='normalized Frobenius',title='Output-gain calibration'); style_axes(ax)
    save_figure(fig,4,'output-gain-calibration','Output-gain calibration',[str(gain_path)],'Does explicit output scaling provide a valid phase control?','Gain rises only weakly because normalization changes the asymptotic operator response.','This negative calibration shows output scale is not a clean control.')

    eps_path=Path('/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity/processed/epsilon_sensitivity_summary.csv'); eps=read_csv(eps_path).sort_values('epsilon')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); axes[0].errorbar(eps.epsilon,eps.frobenius_mean,yerr=eps.frobenius_std,marker='o',capsize=4); axes[1].errorbar(eps.epsilon,eps.nearby_log_growth_mean,yerr=eps.nearby_log_growth_std,marker='o',capsize=4)
    for a in axes: a.set_xscale('log'); a.set_xlabel('perturbation epsilon'); style_axes(a)
    axes[0].set(ylabel='Frobenius',title='AD metric invariance'); axes[1].axhline(0,color='#555',ls='--'); axes[1].set(ylabel='nearby log growth',title='Finite-difference sensitivity'); fig.tight_layout()
    save_figure(fig,4,'epsilon-sensitivity','Epsilon sensitivity',[str(eps_path)],'Which diagnostics are numerically stable in float32?','Frobenius is invariant while nearby trajectory estimates become noisy at small epsilon.','Nearby sign at order 1e-4 per step is not reliable.')


def phase5():
    lyap_root=Path('/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement'); path=lyap_root/'processed/paper_aligned_lyapunov_summary.csv'; d=read_csv(path); raw5=read_jsonl_glob(lyap_root,'alpha*/raw/*dynamical_edge.jsonl')
    fig,axes=plt.subplots(1,3,figsize=(14,4.2)); metrics=[('lyapunov_mean','maximal Lyapunov'),('frobenius_mean','normalized Frobenius'),('final_to_initial_mean','final / initial separation')]
    raw_metrics=['maximal_lyapunov_mean','normalized_frobenius_geomean','final_to_initial_separation']
    for ax,(col,label),raw_col in zip(axes,metrics,raw_metrics):
        ax.plot(d.residual_alpha,d[col],marker='o',color='#3976b9')
        for alpha,g in raw5.groupby('residual_alpha'): ax.scatter(np.full(len(g),alpha),g[raw_col],s=22,color='#3976b9',alpha=.45)
        ax.set(xlabel='residual alpha',ylabel=label); style_axes(ax)
    axes[0].axhline(0,color='#555',ls='--'); axes[1].axhline(1,color='#555',ls='--'); axes[2].axhline(1,color='#555',ls='--'); fig.suptitle('Paper-aligned residual calibration'); fig.tight_layout()
    save_figure(fig,5,'paper-aligned-residual-calibration','Paper-aligned residual calibration',[str(path),str(lyap_root/'alpha*/raw/*dynamical_edge.jsonl')],'Do corrected diagnostics recover identity and contraction controls?','Identity gives Lyapunov approximately 0, Frobenius=1, separation=1; increasing alpha moves into contraction.','This validates measurement mechanics, not the scientific operator choice.')
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(d.frobenius_mean,d.lyapunov_mean,s=70,color='#3976b9');
    for _,r in d.iterrows(): ax.annotate(f"α={r.residual_alpha:g}",(r.frobenius_mean,r.lyapunov_mean),xytext=(5,5),textcoords='offset points')
    ax.axvline(1,color='#555',ls='--'); ax.axhline(0,color='#555',ls='--'); ax.set(xlabel='normalized Frobenius',ylabel='maximal Lyapunov',title='Average gain vs leading tangent growth'); style_axes(ax)
    save_figure(fig,5,'frobenius-vs-lyapunov-calibration','Frobenius versus Lyapunov calibration',[str(path)],'Do the two phase diagnostics agree on controlled operators?','They agree on identity and contraction controls.', 'Agreement on controls does not guarantee equivalence for structured Transformer Jacobians.')


def phase6():
    root=Path('/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality'); long_root=root/'long_asymptotic'
    perf_path=long_root/'processed/long_checkpoint_criticality_performance.csv'; perf=read_csv(perf_path).sort_values('training_step')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(perf.training_step,perf.token_weighted_loss,marker='o',color='#3976b9'); axes[1].plot(perf.training_step,perf.token_weighted_perplexity,marker='o',color='#3976b9'); axes[1].set_yscale('log')
    for a in axes: a.set_xscale('symlog',linthresh=100); a.set_xlabel('training step'); style_axes(a)
    axes[0].set(ylabel='token-weighted loss',title='Validation loss'); axes[1].set(ylabel='perplexity (log)',title='Validation perplexity'); fig.tight_layout()
    save_figure(fig,6,'checkpoint-performance','Checkpoint performance',[str(perf_path)],'Does pretraining improve language-model performance?', 'Loss and perplexity improve sharply and then saturate.', 'Only four checkpoints are shown.')

    raw=read_jsonl_glob(long_root/'raw','*__dynamical_edge.jsonl'); raw['training_step']=raw.checkpoint.map(CHECKPOINT_STEP)
    fig,ax=plt.subplots(figsize=(8,5));
    for checkpoint in CHECKPOINT_ORDER:
        g=raw[raw.checkpoint==checkpoint]; x=CHECKPOINT_STEP[checkpoint]; status=CHECKPOINT_STATUS[checkpoint]; marker={'chaotic':'^','unresolved':'D','stable':'o'}[status]; ax.scatter(np.full(len(g),x),g.maximal_lyapunov_mean,color=PHASE_COLORS[status],marker=marker,alpha=.8); ax.errorbar(x,g.maximal_lyapunov_mean.mean(),yerr=g.maximal_lyapunov_mean.std(),color=PHASE_COLORS[status],capsize=5,fmt='none')
    ax.axhline(0,color='#555',ls='--'); ax.set_xscale('symlog',linthresh=100); ax.set(xlabel='training step',ylabel='maximal Lyapunov',title='Sample-level Lyapunov across training'); style_axes(ax)
    save_figure(fig,6,'checkpoint-lyapunov-samples','Checkpoint Lyapunov samples',[str(long_root/'raw/*dynamical_edge.jsonl')],'Does training move the feedback operator toward zero Lyapunov?','The operator moves from positive/near-zero to strongly negative Lyapunov.', 'step1000 remains phase-unresolved; this is a constructed feedback operator.')

    fig,ax=plt.subplots(figsize=(6,5))
    for _,r in perf.iterrows():
        status=CHECKPOINT_STATUS[r.checkpoint]; ax.scatter(r.frobenius_mean,r.maximal_lyapunov_mean,color=PHASE_COLORS[status],marker={'chaotic':'^','unresolved':'D','stable':'o'}[status],s=80); ax.annotate(r.checkpoint,(r.frobenius_mean,r.maximal_lyapunov_mean),xytext=(5,5),textcoords='offset points')
    ax.axvline(1,color='#555',ls='--'); ax.axhline(0,color='#555',ls='--'); ax.set(xlabel='normalized Frobenius',ylabel='maximal Lyapunov',title='Frobenius proxy failure at step0'); style_axes(ax)
    save_figure(fig,6,'checkpoint-frobenius-vs-lyapunov','Checkpoint Frobenius versus Lyapunov',[str(perf_path)],'Does Frobenius correctly classify Transformer phase?','step0 has Frobenius<1 but positive Lyapunov, providing a direct counterexample to proxy equivalence.', 'This refutes the proxy for this operator, not the original theory in all systems.')

    comp_path=long_root/'processed/short_vs_long_comparison.csv'; comp=read_csv(comp_path)
    fig,ax=plt.subplots(figsize=(9,4.8)); width=.35; xs=np.arange(4)
    for i,(protocol,g) in enumerate(comp.groupby('protocol')):
        g=g.set_index('checkpoint').loc[CHECKPOINT_ORDER]; ax.bar(xs+(i-.5)*width,g.tail_relative_step_delta_mean,width,label=protocol)
    ax.set_yscale('log'); ax.set_xticks(xs,CHECKPOINT_ORDER); ax.set(ylabel='tail relative step delta (log)',title='Short versus long asymptotic protocol'); ax.legend(fontsize=8); style_axes(ax)
    save_figure(fig,6,'short-vs-long-protocol','Short versus long burn-in',[str(comp_path)],'How much does burn-in alter convergence assessment?','Long burn-in is essential for early checkpoints and confirms only the final checkpoint as a strict fixed point.', 'Non-fixed chaotic attractors should not be rejected solely by fixed-point convergence.')

    traj=read_jsonl_glob(long_root/'raw','*__dynamics_trajectory.jsonl')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,checkpoint in zip(axes.flat,CHECKPOINT_ORDER):
        status=CHECKPOINT_STATUS[checkpoint]
        for _,g in traj[traj.checkpoint==checkpoint].groupby('sample_index'): ax.plot(g.step_index,g.relative_step_delta,alpha=.65,color=PHASE_COLORS[status],marker=PHASE_MARKERS[status],markevery=32,markersize=3)
        ax.set_yscale('log'); ax.set_title(checkpoint); ax.set_ylabel('relative step delta'); style_axes(ax)
    axes[-1,0].set_xlabel('eval step'); axes[-1,1].set_xlabel('eval step'); fig.suptitle('Convergence trajectories by checkpoint'); fig.tight_layout()
    save_figure(fig,6,'checkpoint-relative-step-trajectories','Relative-step trajectories',[str(long_root/'raw/*dynamics_trajectory.jsonl')],'Which checkpoints approach fixed points?', 'step143000 collapses to numerical-scale step changes; step16000 converges slowly; early checkpoints remain non-fixed.', 'Non-fixed behavior needs Lyapunov and recurrence evidence for phase classification.')

    lag_path=long_root/'processed/long_recurrence_lag_summary.csv'; lag=read_csv(lag_path); pivot=lag.pivot(index='checkpoint',columns='lag_window',values='normalized_lag_distance_mean').reindex(CHECKPOINT_ORDER)
    fig,ax=plt.subplots(figsize=(9,4)); im=ax.imshow(np.log10(pivot.clip(lower=1e-10)),aspect='auto',cmap='viridis'); ax.set_yticks(range(len(pivot)),pivot.index); ax.set_xticks(range(len(pivot.columns)),pivot.columns); ax.set(xlabel='lag window',ylabel='checkpoint',title='Normalized recurrence distance'); fig.colorbar(im,ax=ax,label='log10 normalized lag distance'); fig.tight_layout()
    save_figure(fig,6,'checkpoint-recurrence-heatmap','Checkpoint recurrence heatmap',[str(lag_path)],'Do trajectories recur over short lags?', 'Late checkpoints have tiny recurrence distances; early checkpoints show no recurrence up to lag64.', 'Absence of short-lag recurrence does not prove chaos without Lyapunov/stationarity evidence.')

    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,checkpoint in zip(axes.flat,CHECKPOINT_ORDER):
        status=CHECKPOINT_STATUS[checkpoint]
        for _,g in traj[traj.checkpoint==checkpoint].groupby('sample_index'): ax.plot(g.step_index,g.nearby_distance,alpha=.65,color=PHASE_COLORS[status],marker=PHASE_MARKERS[status],markevery=32,markersize=3)
        ax.set_yscale('log'); ax.set_title(checkpoint); ax.set_ylabel('nearby distance'); style_axes(ax)
    axes[-1,0].set_xlabel('eval step'); axes[-1,1].set_xlabel('eval step'); fig.suptitle('Nearby trajectory separation'); fig.tight_layout()
    save_figure(fig,6,'checkpoint-nearby-trajectories','Checkpoint nearby trajectories',[str(long_root/'raw/*dynamics_trajectory.jsonl')],'Do small perturbations expand or contract?', 'step0 expands strongly; late checkpoints contract.', 'Finite distances can saturate and are secondary to renormalized Lyapunov.')

    visual_root=root/'visualization_rerun/raw'; visual=read_jsonl_glob(visual_root,'*__dynamics_trajectory.jsonl')
    if not visual.empty:
        point_rows=[]
        for (checkpoint,sample),g in visual.groupby(['checkpoint','sample_index']): point_rows.extend(projected_poincare_crossings(g.to_dict('records')))
        points=pd.DataFrame(point_rows); processed=root/'visualization_rerun/processed'; processed.mkdir(parents=True,exist_ok=True); visual.to_csv(processed/'projection_trajectory_rows.csv',index=False); points.to_csv(processed/'projected_poincare_points.csv',index=False)
        fig=plt.figure(figsize=(12,10))
        for idx,checkpoint in enumerate(CHECKPOINT_ORDER,1):
            ax=fig.add_subplot(2,2,idx,projection='3d')
            g=visual[(visual.checkpoint==checkpoint)&(visual.sample_index==0)].sort_values('step_index')
            status=CHECKPOINT_STATUS[checkpoint]
            z=g[['projection_0','projection_1','projection_2']].to_numpy(float); relative=z-z[-1]
            ax.plot(relative[:,0],relative[:,1],relative[:,2],color=PHASE_COLORS[status],linewidth=1,alpha=.85)
            ax.scatter(relative[::32,0],relative[::32,1],relative[::32,2],c=g.step_index.to_numpy()[::32],cmap='viridis',s=13)
            ax.scatter(*relative[0],marker='x',s=35,color='#222'); ax.scatter(0,0,0,marker='*',s=55,color='#111')
            span=float(np.linalg.norm(np.ptp(z,axis=0))); ax.set_title(f'{checkpoint}, sample0; span={span:.2e}')
            ax.set_xlabel('Δz0 from final'); ax.set_ylabel('Δz1 from final'); ax.set_zlabel('Δz2 from final')
        fig.suptitle('Matched sample0 trajectories, centered at the final projected state'); fig.tight_layout()
        save_figure(fig,6,'checkpoint-three-projection-trajectories','Matched sample0 three-projection trajectories',[str(visual_root/'*dynamics_trajectory.jsonl')],'How does the same sample0 projected trajectory approach its final state?', 'Centering at the final projected state exposes the step16000 contraction and the numerical-scale step143000 jitter.', 'The star is the final projected state, not proof of a globally unique fixed point; three projections can hide high-dimensional structure.')
        fig,axes=plt.subplots(2,2,figsize=(10,9))
        for ax,checkpoint in zip(axes.flat,CHECKPOINT_ORDER):
            status=CHECKPOINT_STATUS[checkpoint]; g=visual[(visual.checkpoint==checkpoint)&(visual.sample_index==0)].dropna(subset=['projection_0_next']).sort_values('step_index'); final=float(g.projection_0.iloc[-1]); ax.scatter(g.projection_0-final,g.projection_0_next-final,s=10,alpha=.55,color=PHASE_COLORS[status],marker=PHASE_MARKERS[status]); ax.scatter(0,0,marker='*',s=55,color='#111'); ax.set(title=f'{checkpoint}, sample0',xlabel='Δz0(t) from final',ylabel='Δz0(t+1) from final'); style_axes(ax)
        fig.suptitle('Matched sample0 centered return maps'); fig.tight_layout()
        save_figure(fig,6,'checkpoint-return-maps','Matched sample0 centered return maps',[str(visual_root/'*dynamics_trajectory.jsonl')],'Does the same projected coordinate approach a fixed return point or remain extended?', 'Late checkpoints compress toward the origin; early checkpoints remain extended.', 'A one-coordinate return map is projection-dependent and step143000 residual spread is float32-scale jitter.')
        fig,axes=plt.subplots(2,2,figsize=(10,9))
        for ax,checkpoint in zip(axes.flat,CHECKPOINT_ORDER):
            g=points[(points.checkpoint==checkpoint)&(points.sample_index==0)].sort_values('crossing_order') if not points.empty else pd.DataFrame();
            if g.empty: ax.text(.5,.5,'No upward crossings',ha='center',va='center',transform=ax.transAxes)
            else:
                vg=visual[(visual.checkpoint==checkpoint)&(visual.sample_index==0)].sort_values('step_index'); final_z1=float(vg.projection_1.iloc[-1]); final_z2=float(vg.projection_2.iloc[-1]); x=g.poincare_z1-final_z1; y=g.poincare_z2-final_z2; ax.plot(x,y,color='#aaa',linewidth=.7); sc=ax.scatter(x,y,c=g.crossing_order,cmap='viridis',s=28); ax.scatter(0,0,marker='*',s=55,color='#111')
            ax.set(title=f'{checkpoint}, sample0; n={len(g)}',xlabel='Δz1 at z0 crossing',ylabel='Δz2 at z0 crossing'); style_axes(ax)
        fig.suptitle('Matched sample0 Projected Poincaré Sections (purple early → yellow late)'); fig.tight_layout()
        save_figure(fig,6,'checkpoint-projected-poincare','Matched sample0 projected Poincare sections',[str(processed/'projected_poincare_points.csv')],'Do crossings of the same sample0 trajectory contract toward its final projected state?', 'step16000 crossings contract toward the final state; step143000 crossings are confined to numerical-scale jitter.', 'Each trajectory uses its own median-z0 section. A fixed point may yield no true crossings; repeated late crossings can be roundoff jitter, not a period.')

        fig,axes=plt.subplots(2,2,figsize=(10,9))
        sample_colors=plt.get_cmap('tab10')
        for ax,checkpoint in zip(axes.flat,CHECKPOINT_ORDER):
            for sample,g in points[points.checkpoint==checkpoint].groupby('sample_index'):
                g=g.sort_values('crossing_order'); vg=visual[(visual.checkpoint==checkpoint)&(visual.sample_index==sample)].sort_values('step_index'); final_z1=float(vg.projection_1.iloc[-1]); final_z2=float(vg.projection_2.iloc[-1]); x=g.poincare_z1-final_z1; y=g.poincare_z2-final_z2; ax.plot(x,y,linewidth=.7,alpha=.6,color=sample_colors(int(sample)%10)); ax.scatter(x,y,s=12,alpha=.65,color=sample_colors(int(sample)%10),label=f's{int(sample)}')
            ax.scatter(0,0,marker='*',s=50,color='#111'); ax.set(title=checkpoint,xlabel='Δz1 from sample final',ylabel='Δz2 from sample final'); style_axes(ax)
        axes[0,0].legend(ncol=2,fontsize=7); fig.suptitle('Per-sample centered Projected Poincaré diagnostics'); fig.tight_layout()
        save_figure(fig,6,'checkpoint-poincare-per-sample-centered','Per-sample centered projected Poincare diagnostics',[str(processed/'projected_poincare_points.csv')],'Do all eight samples show within-trajectory contraction after removing different fixed-point locations?', 'Centering separates within-sample contraction from the much larger between-sample fixed-point offsets.', 'This overlays eight different median-z0 sections after centering; it is a diagnostic comparison, not one common Poincare section.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for fn in (phase2,phase3,phase4,phase5,phase6): fn()
    frame=pd.DataFrame(manifest); frame.to_csv(OUT/'figure_manifest.csv',index=False)
    lines=['# Experiment Visualization Figure Manifest','']
    for phase,g in frame.groupby('phase'):
        lines += [f'## Phase {phase}','']
        for _,r in g.iterrows(): lines += [f"### {r['title']}",'',f"- Figure: `{r['full_path']}`",f"- Curated: `{r['curated_path']}`",f"- Sources: {r['sources']}",f"- Checkpoints/models: {r['checkpoints_models']}",f"- Sample count: {r['sample_count']}",f"- Filtering: {r['filtering']}",f"- Axes and units: {r['axes_units']}",f"- Question: {r['question']}",f"- Interpretation: {r['interpretation']}",f"- Caveat: {r['caveat']}",f"- Evidence: {r['evidence_status']}",'']
    (OUT/'figure_manifest.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'wrote {len(frame)} figures under {OUT}')


if __name__ == '__main__': main()
