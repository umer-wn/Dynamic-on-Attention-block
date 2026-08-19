#!/usr/bin/env python3
"""Build projection-1 x_t vs x_{t+1} phase portraits from Experiment 16.

Uses the last 512 transitions for every checkpoint and token series.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EARLY_SOURCE = ROOT / "experiments_ordered/16_frequency_stratified_window_jacobian/processed/projection_trajectory.csv"
DENSE_SOURCE = ROOT / "experiments_ordered/25_dense_checkpoint_suite/processed/state_projection_trajectory.csv"
EXP = ROOT / "experiments_ordered/17_visualize"
PROCESSED = EXP / "processed"
FIGURES = EXP / "figures/projection1_xt_xtplus1_last512"
OUT_CSV = PROCESSED / "projection1_xt_xtplus1_last512.csv"
OUT_HTML = EXP / "projection1_xt_xtplus1_last512.html"
OUT_GRID = EXP / "figures/projection1_xt_xtplus1_last512_grid.png"
PERTURBATION_CSV = PROCESSED / "single_token_perturbation_projection_trajectories.csv"
PLOTLY = EXP / "plotly-2.35.2.min.js"

LAST_TRANSITIONS = 512
COLORS = ["#2878B5", "#D95F02", "#2A9D8F", "#8E5BB7", "#C44E52", "#4C956C", "#E9C46A", "#577590"]


def checkpoint_number(name: str) -> int:
    return int(name.removeprefix("step"))


def source_rows():
    """Yield the dashboard's complete 59-checkpoint, four-token state set."""
    with EARLY_SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = checkpoint_number(row["checkpoint"])
            if 0 < step <= 10000:
                yield row
    with DENSE_SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def load_series():
    grouped: dict[tuple[str, int], dict] = {}
    for row in source_rows():
        key = (row["checkpoint"], int(row["selection_index"]))
        item = grouped.setdefault(
            key,
            {
                "checkpoint": row["checkpoint"],
                "selection_index": int(row["selection_index"]),
                "token_id": int(row["token_id"]),
                "token": row["token"],
                "frequency_bin": int(row["frequency_bin"]),
                "steps": [],
                "values": [],
            },
        )
        item["steps"].append(int(row["dynamic_step"]))
        item["values"].append(float(row["projection_1"]))

    output = []
    for item in grouped.values():
        order = np.argsort(item["steps"])
        steps = np.asarray(item["steps"], dtype=int)[order]
        values = np.asarray(item["values"], dtype=float)[order]
        if len(values) < LAST_TRANSITIONS + 1:
            raise ValueError(f"{item['checkpoint']} / token {item['token_id']} has only {len(values)} states")
        steps = steps[-(LAST_TRANSITIONS + 1) :]
        values = values[-(LAST_TRANSITIONS + 1) :]
        if not np.all(np.diff(steps) == 1):
            raise ValueError(f"Non-consecutive dynamic steps in {item['checkpoint']} / token {item['token_id']}")
        item["t"] = steps[:-1]
        item["t_plus_1"] = steps[1:]
        item["x_t"] = values[:-1]
        item["x_t_plus_1"] = values[1:]
        output.append(item)

    output.sort(key=lambda d: (checkpoint_number(d["checkpoint"]), d["selection_index"]))
    return output


def write_csv(series):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "selection_index",
        "token_id",
        "token",
        "frequency_bin",
        "dynamic_step_t",
        "dynamic_step_t_plus_1",
        "projection_1_x_t",
        "projection_1_x_t_plus_1",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in series:
            for t, tp1, x, y in zip(item["t"], item["t_plus_1"], item["x_t"], item["x_t_plus_1"]):
                writer.writerow(
                    {
                        "checkpoint": item["checkpoint"],
                        "selection_index": item["selection_index"],
                        "token_id": item["token_id"],
                        "token": item["token"],
                        "frequency_bin": item["frequency_bin"],
                        "dynamic_step_t": int(t),
                        "dynamic_step_t_plus_1": int(tp1),
                        "projection_1_x_t": f"{x:.10g}",
                        "projection_1_x_t_plus_1": f"{y:.10g}",
                    }
                )


def axis_bounds(items):
    vals = np.concatenate([np.concatenate([d["x_t"], d["x_t_plus_1"]]) for d in items])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = max((hi - lo) * 0.06, 1e-9)
    return lo - pad, hi + pad


def draw_checkpoint(ax, checkpoint, items, compact=False):
    lo, hi = axis_bounds(items)
    ax.plot([lo, hi], [lo, hi], color="#A0A0A0", lw=0.8, ls="--", zorder=0)
    for idx, item in enumerate(items):
        color = COLORS[idx % len(COLORS)]
        label = f"{item['token']} (id={item['token_id']})"
        ax.plot(item["x_t"], item["x_t_plus_1"], color=color, lw=0.8, alpha=0.62, label=label)
        ax.scatter(item["x_t"][0], item["x_t_plus_1"][0], s=16, marker="o", color=color, edgecolor="white", linewidth=0.5)
        ax.scatter(item["x_t"][-1], item["x_t_plus_1"][-1], s=24, marker="X", color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title(checkpoint, fontsize=9 if compact else 12)
    ax.set_xlabel(r"projection 1: $x_t$", fontsize=7 if compact else 10)
    ax.set_ylabel(r"projection 1: $x_{t+1}$", fontsize=7 if compact else 10)
    ax.tick_params(labelsize=6 if compact else 8)


def write_figures(series):
    FIGURES.mkdir(parents=True, exist_ok=True)
    by_checkpoint: dict[str, list] = {}
    for item in series:
        by_checkpoint.setdefault(item["checkpoint"], []).append(item)
    checkpoints = sorted(by_checkpoint, key=checkpoint_number)

    for ck in checkpoints:
        fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
        draw_checkpoint(ax, ck, by_checkpoint[ck])
        ax.legend(loc="best", fontsize=7, frameon=True)
        fig.suptitle("Projection-1 return map · last 512 transitions", fontsize=13)
        fig.savefig(FIGURES / f"projection1_xt_xtplus1_last512_{ck}.png", dpi=180)
        plt.close(fig)

    cols = 5
    rows = int(np.ceil(len(checkpoints) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, ck in zip(axes, checkpoints):
        draw_checkpoint(ax, ck, by_checkpoint[ck], compact=True)
    for ax in axes[len(checkpoints) :]:
        ax.axis("off")
    fig.suptitle("Projection-1 return maps by checkpoint · last 512 transitions", fontsize=16)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=160)
    plt.close(fig)
    return by_checkpoint, checkpoints


def load_full_projection_payload(checkpoints):
    grouped: dict[tuple[str, int], dict] = {}
    for row in source_rows():
        key = (row["checkpoint"], int(row["selection_index"]))
        item = grouped.setdefault(
            key,
            {
                "selection_index": int(row["selection_index"]),
                "token_id": int(row["token_id"]),
                "token": row["token"],
                "frequency_bin": int(row["frequency_bin"]),
                "steps": [],
                "p1": [],
                "p2": [],
                "p3": [],
                "p4": [],
            },
        )
        item["steps"].append(int(row["dynamic_step"]))
        for projection in range(1, 5):
            item[f"p{projection}"].append(round(float(row[f"projection_{projection}"]), 8))

    payload = {"checkpoints": checkpoints, "data": {ck: [] for ck in checkpoints}}
    for (checkpoint, _), item in sorted(
        grouped.items(), key=lambda pair: (checkpoint_number(pair[0][0]), pair[0][1])
    ):
        order = np.argsort(item["steps"])
        packed = {
            "selection_index": item["selection_index"],
            "token_id": item["token_id"],
            "token": item["token"],
            "frequency_bin": item["frequency_bin"],
            "steps": np.asarray(item["steps"], dtype=int)[order].tolist(),
        }
        for projection in range(1, 5):
            packed[f"p{projection}"] = np.asarray(item[f"p{projection}"], dtype=float)[order].tolist()
        payload["data"][checkpoint].append(packed)
    if not PERTURBATION_CSV.exists():
        raise FileNotFoundError(f"missing perturbation experiment output: {PERTURBATION_CSV}")
    perturbation = {"data": {ck: None for ck in checkpoints}}
    perturbation_rows: dict[str, list[dict]] = {ck: [] for ck in checkpoints}
    with PERTURBATION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["checkpoint"] in perturbation_rows:
                perturbation_rows[row["checkpoint"]].append(row)
    for checkpoint in checkpoints:
        rows = sorted(perturbation_rows[checkpoint], key=lambda row: int(row["dynamic_step"]))
        if not rows:
            continue
        packed = {
            "selection_index": int(rows[0]["selection_index"]),
            "token_id": int(rows[0]["token_id"]),
            "token": rows[0]["token"],
            "relative_scale": float(rows[0]["relative_scale"]),
            "perturbation_seed": int(rows[0]["perturbation_seed"]),
            "steps": [int(row["dynamic_step"]) for row in rows],
            "distance": [round(float(row["full_state_distance"]), 8) for row in rows],
        }
        for projection in range(1, 5):
            packed[f"o{projection}"] = [
                round(float(row[f"original_projection_{projection}"]), 8) for row in rows
            ]
            packed[f"q{projection}"] = [
                round(float(row[f"perturbed_projection_{projection}"]), 8) for row in rows
            ]
        perturbation["data"][checkpoint] = packed
    first_perturbation = next((perturbation["data"][ck] for ck in checkpoints if perturbation["data"][ck]), None)
    perturbation["metadata"] = ({
        "selection_index": first_perturbation["selection_index"],
        "token_id": first_perturbation["token_id"],
        "token": first_perturbation["token"],
        "relative_scale": first_perturbation["relative_scale"],
        "perturbation_seed": first_perturbation["perturbation_seed"],
    } if first_perturbation else None)
    payload["perturbation"] = perturbation
    return payload


def write_html(by_checkpoint, checkpoints):
    payload = load_full_projection_payload(checkpoints)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paired state and return-map trajectories · all checkpoints</title>
<script>__PLOTLY__</script>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f6f7f9;color:#1d232d}}
.wrap{{max-width:1700px;margin:0 auto;padding:18px}} h1{{font-size:22px;margin:0 0 7px}} .sub{{color:#667085;margin-bottom:12px}}
.controls{{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;background:white;padding:12px 16px;border:1px solid #e1e5eb;border-radius:10px}}
.control{{display:flex;flex-direction:column;gap:7px;min-width:150px}} .control-title{{font-size:13px;font-weight:600;color:#344054}}
select{{font-size:14px;padding:6px 10px}} .tokens{{display:flex;gap:12px;flex-wrap:wrap}} label{{font-size:14px;white-space:nowrap}}
.ranges{{display:grid;grid-template-columns:auto 210px auto;gap:7px 10px;align-items:center}} input[type=range]{{width:210px}}
.range-value{{font-variant-numeric:tabular-nums;color:#344054;font-size:13px}} #plot{{background:white;margin-top:14px;border:1px solid #e1e5eb;border-radius:10px}}
.time-key{{display:flex;align-items:center;gap:8px;font-size:13px;color:#475467}} .time-ramp{{width:120px;height:9px;background:linear-gradient(90deg,rgba(40,120,181,.05),rgba(40,120,181,1));border-radius:5px}}
.hint{{font-size:12px;color:#667085;margin-top:8px}} @media(max-width:700px){{.wrap{{padding:10px}} .ranges{{grid-template-columns:auto 1fr auto}} input[type=range]{{width:100%}}}}
</style></head><body><div class="wrap">
<h1>State projections + return maps</h1>
<div class="sub">59个checkpoint同时呈现；左侧为可选二维状态投影，右侧为可选方向 x<sub>t</sub>–x<sub>t+1</sub> return map。默认最后512步，可扩展到0–1024。</div>
<div class="controls">
  <div class="control"><div class="control-title">回归相图方向</div><select id="projection"><option value="1">Projection 1</option><option value="2">Projection 2</option><option value="3">Projection 3</option><option value="4">Projection 4</option></select></div>
  <div class="control"><div class="control-title">二维投影平面</div><select id="projectionPair"><option value="1,2">P1 × P2</option><option value="1,3">P1 × P3</option><option value="1,4">P1 × P4</option><option value="2,3">P2 × P3</option><option value="2,4">P2 × P4</option><option value="3,4">P3 × P4</option></select></div>
  <div class="control"><div class="control-title">Token</div><div class="tokens" id="tokens"></div></div>
  <div class="control"><div class="control-title">Dynamic-step 范围 <span class="range-value" id="rangeSummary"></span></div><div class="ranges"><span>起点</span><input id="startStep" type="range" min="0" max="1023" value="512" step="1"><span class="range-value" id="startValue">512</span><span>终点</span><input id="endStep" type="range" min="1" max="1024" value="1024" step="1"><span class="range-value" id="endValue">1024</span></div></div>
  <div class="control"><div class="control-title">时间编码</div><div class="time-key"><span>浅</span><span class="time-ramp"></span><span>深</span></div><div class="range-value" id="perturbInfo"></div></div>
</div>
<div id="plot" role="img" aria-label="所有 checkpoint 的二维状态投影与 return-map 配对图"></div><div class="hint">深色表示更大的 t。微扰 token 的原始终点为圆形，微扰终点为 X；终点随所选 dynamic-step 终点同步更新。</div>
</div><script>
const payload={data_json};
const colors={json.dumps(COLORS)};
const tokenSeries=payload.data[payload.checkpoints[0]];
const tokenBox=document.getElementById('tokens');
tokenSeries.forEach((s,i)=>{{
 const label=document.createElement('label');
 label.innerHTML=`<input type="checkbox" value="${{s.selection_index}}" checked> ${{s.token}} <span style="color:#667085">(id=${{s.token_id}})</span>`;
 tokenBox.appendChild(label);
}});
document.getElementById('perturbInfo').textContent=payload.perturbation.metadata?`微扰（有数据的checkpoint）：${{payload.perturbation.metadata.token}} · relative scale=${{payload.perturbation.metadata.relative_scale}}`:'无内置微扰轨迹';
function selectedTokens(){{return new Set([...tokenBox.querySelectorAll('input:checked')].map(el=>Number(el.value)));}}
function axisName(prefix,index){{return index===0?prefix:`${{prefix}}${{index+1}}`;}}
function hexRgb(hex){{const h=hex.replace('#','');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}}
function gradient(hex){{const [r,g,b]=hexRgb(hex);return [[0,`rgba(${{r}},${{g}},${{b}},0.04)`],[0.35,`rgba(${{r}},${{g}},${{b}},0.22)`],[1,`rgba(${{r}},${{g}},${{b}},1)`]];}}
function rgba(hex,a){{const [r,g,b]=hexRgb(hex);return `rgba(${{r}},${{g}},${{b}},${{a}})`;}}
function rowDomain(index,rows){{const gap=0.012,h=(1-gap*(rows-1))/rows;return [1-(index+1)*h-index*gap,1-index*h-index*gap];}}
function bounds(values){{if(!values.length)return [-1,1];let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*0.07,1e-8);return [lo-pad,hi+pad];}}
function indexForStep(steps,target){{let idx=steps.indexOf(target);if(idx>=0)return idx;idx=steps.findIndex(v=>v>target);return idx<0?steps.length-1:Math.max(0,idx-1);}}
function addEndpoint(traces,x,y,xaxis,yaxis,name,color,symbol,custom,showlegend){{
 traces.push({{type:'scatter',x:[x],y:[y],xaxis,yaxis,mode:'markers',name,legendgroup:name,showlegend,marker:{{color,size:10,symbol,line:{{color:'#ffffff',width:1.5}}}},customdata:[custom],hovertemplate:'%{{customdata[0]}}<br>checkpoint=%{{customdata[1]}}<br>t=%{{customdata[2]}}<br>x=%{{x:.7g}}<br>y=%{{y:.7g}}<br>full-state distance=%{{customdata[3]:.6g}}<extra></extra>'}});
}}
function updateRangeLabels(){{
 const start=document.getElementById('startStep'),end=document.getElementById('endStep');
 let a=Number(start.value),b=Number(end.value);
 if(a>=b){{if(document.activeElement===start) b=Math.min(1024,a+1); else a=Math.max(0,b-1); start.value=a; end.value=b;}}
 document.getElementById('startValue').textContent=a; document.getElementById('endValue').textContent=b;
 document.getElementById('rangeSummary').textContent=`${{a}}–${{b}} · ${{b-a}} transitions`;
 return [a,b];
}}
function render(){{
 const projection=Number(document.getElementById('projection').value); const [projectionA,projectionB]=document.getElementById('projectionPair').value.split(',').map(Number); const selected=selectedTokens(); const [start,end]=updateRangeLabels();
 const rows=payload.checkpoints.length,traces=[],shapes=[],annotations=[]; const layout={{margin:{{l:68,r:35,t:90,b:90}},height:rows*330+120,showlegend:true,legend:{{orientation:'h',x:0.5,xanchor:'center',y:1.012,yanchor:'bottom'}},hovermode:'closest',paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',shapes,annotations}};
 payload.checkpoints.forEach((ck,ci)=>{{
   const yDomain=rowDomain(ci,rows),leftIndex=ci*2,rightIndex=ci*2+1,leftX=axisName('x',leftIndex),leftY=axisName('y',leftIndex),rightX=axisName('x',rightIndex),rightY=axisName('y',rightIndex); const leftAll=[],rightAll=[];
   const active=payload.data[ck].filter(s=>selected.has(s.selection_index));
   active.forEach((s,si)=>{{
     const pa=s[`p${{projectionA}}`],pb=s[`p${{projectionB}}`],pr=s[`p${{projection}}`],steps=s.steps,base=colors[s.selection_index%colors.length]; const lx=[],ly=[],lt=[],lc=[],rx=[],ry=[],rt=[],rc=[];
     for(let i=0;i<steps.length;i++){{const t=steps[i];if(t<start||t>end)continue;lx.push(pa[i]);ly.push(pb[i]);lt.push(t);lc.push([ck,s.token,s.token_id,t]);if(i<steps.length-1&&t<end){{rx.push(pr[i]);ry.push(pr[i+1]);rt.push(t);rc.push([ck,s.token,s.token_id,t]);}}}}
     leftAll.push(...lx,...ly);rightAll.push(...rx,...ry);
     traces.push({{type:'scattergl',x:lx,y:ly,xaxis:leftX,yaxis:leftY,mode:'lines+markers',name:`${{s.token}} · id=${{s.token_id}}`,legendgroup:`token-${{s.selection_index}}`,showlegend:ci===0,line:{{color:rgba(base,.16),width:1}},marker:{{color:lt,colorscale:gradient(base),cmin:start,cmax:end,size:4,showscale:false}},customdata:lc,hovertemplate:'checkpoint=%{{customdata[0]}}<br>token=%{{customdata[1]}} (id=%{{customdata[2]}})<br>t=%{{customdata[3]}}<br>P${{projectionA}}=%{{x:.7g}}<br>P${{projectionB}}=%{{y:.7g}}<extra></extra>'}});
     traces.push({{type:'scattergl',x:rx,y:ry,xaxis:rightX,yaxis:rightY,mode:'lines+markers',name:`${{s.token}} · id=${{s.token_id}}`,legendgroup:`token-${{s.selection_index}}`,showlegend:false,line:{{color:rgba(base,.16),width:1}},marker:{{color:rt,colorscale:gradient(base),cmin:start,cmax:Math.max(start+1,end-1),size:4,showscale:false}},customdata:rc,hovertemplate:'checkpoint=%{{customdata[0]}}<br>token=%{{customdata[1]}} (id=%{{customdata[2]}})<br>t=%{{customdata[3]}}<br>x_t=%{{x:.7g}}<br>x_t+1=%{{y:.7g}}<extra></extra>'}});
   }});
   const perturb=payload.perturbation.data[ck];
   if(perturb&&selected.has(perturb.selection_index)){{
     const pSteps=perturb.steps,oiStart=indexForStep(pSteps,start),oiEnd=indexForStep(pSteps,end),oA=perturb[`o${{projectionA}}`],oB=perturb[`o${{projectionB}}`],oR=perturb[`o${{projection}}`],qA=perturb[`q${{projectionA}}`],qB=perturb[`q${{projectionB}}`],qR=perturb[`q${{projection}}`],px=[],py=[],pt=[],pc=[],prx=[],pry=[],prt=[],prc=[];
     for(let i=oiStart;i<=oiEnd;i++){{const t=pSteps[i];px.push(qA[i]);py.push(qB[i]);pt.push(t);pc.push([ck,perturb.token,perturb.token_id,t,perturb.distance[i]]);if(i<oiEnd){{prx.push(qR[i]);pry.push(qR[i+1]);prt.push(t);prc.push([ck,perturb.token,perturb.token_id,t,perturb.distance[i+1]]);}}}}
     leftAll.push(...px,...py);rightAll.push(...prx,...pry);
     traces.push({{type:'scattergl',x:px,y:py,xaxis:leftX,yaxis:leftY,mode:'lines+markers',name:`perturbed ${{perturb.token}}`,legendgroup:'perturbed',showlegend:ci===0,line:{{color:'rgba(214,69,65,.2)',width:1}},marker:{{color:pt,colorscale:[[0,'rgba(214,69,65,.05)'],[1,'rgba(214,69,65,1)']],cmin:start,cmax:end,size:4}},customdata:pc,hovertemplate:'PERTURBED<br>checkpoint=%{{customdata[0]}}<br>token=%{{customdata[1]}}<br>t=%{{customdata[3]}}<br>P${{projectionA}}=%{{x:.7g}}<br>P${{projectionB}}=%{{y:.7g}}<br>full-state distance=%{{customdata[4]:.6g}}<extra></extra>'}});
     traces.push({{type:'scattergl',x:prx,y:pry,xaxis:rightX,yaxis:rightY,mode:'lines+markers',name:`perturbed ${{perturb.token}}`,legendgroup:'perturbed',showlegend:false,line:{{color:'rgba(214,69,65,.2)',width:1}},marker:{{color:prt,colorscale:[[0,'rgba(214,69,65,.05)'],[1,'rgba(214,69,65,1)']],cmin:start,cmax:Math.max(start+1,end-1),size:4}},customdata:prc,hovertemplate:'PERTURBED<br>checkpoint=%{{customdata[0]}}<br>token=%{{customdata[1]}}<br>t=%{{customdata[3]}}<br>x_t=%{{x:.7g}}<br>x_t+1=%{{y:.7g}}<br>full-state distance=%{{customdata[4]:.6g}}<extra></extra>'}});
     const endpointCustomOriginal=['original endpoint',ck,end,0],endpointCustomPerturbed=['perturbed endpoint',ck,end,perturb.distance[oiEnd]];
     addEndpoint(traces,oA[oiEnd],oB[oiEnd],leftX,leftY,'original endpoint','#123B5D','circle',endpointCustomOriginal,ci===0); addEndpoint(traces,qA[oiEnd],qB[oiEnd],leftX,leftY,'perturbed endpoint','#D64541','x',endpointCustomPerturbed,ci===0);
     if(oiEnd>0){{addEndpoint(traces,oR[oiEnd-1],oR[oiEnd],rightX,rightY,'original endpoint','#123B5D','circle',endpointCustomOriginal,false);addEndpoint(traces,qR[oiEnd-1],qR[oiEnd],rightX,rightY,'perturbed endpoint','#D64541','x',endpointCustomPerturbed,false);}}
   }}
   const leftRange=bounds(leftAll),rightRange=bounds(rightAll),leftXKey=axisName('xaxis',leftIndex),leftYKey=axisName('yaxis',leftIndex),rightXKey=axisName('xaxis',rightIndex),rightYKey=axisName('yaxis',rightIndex);
   layout[leftXKey]={{domain:[0,0.455],anchor:leftY,range:leftRange,showgrid:true,gridcolor:'#e9edf2',zerolinecolor:'#c6ccd4',tickfont:{{size:9}},title:{{text:`P${{projectionA}}`,font:{{size:10}}}}}};
   layout[leftYKey]={{domain:yDomain,anchor:leftX,range:leftRange,showgrid:true,gridcolor:'#e9edf2',zerolinecolor:'#c6ccd4',tickfont:{{size:9}},scaleanchor:leftX,scaleratio:1,title:{{text:`P${{projectionB}}`,font:{{size:10}}}}}};
   layout[rightXKey]={{domain:[0.545,1],anchor:rightY,range:rightRange,showgrid:true,gridcolor:'#e9edf2',zerolinecolor:'#c6ccd4',tickfont:{{size:9}},title:{{text:`P${{projection}}: x_t`,font:{{size:10}}}}}};
   layout[rightYKey]={{domain:yDomain,anchor:rightX,range:rightRange,showgrid:true,gridcolor:'#e9edf2',zerolinecolor:'#c6ccd4',tickfont:{{size:9}},scaleanchor:rightX,scaleratio:1,title:{{text:`P${{projection}}: x_t+1`,font:{{size:10}}}}}};
   shapes.push({{type:'line',xref:rightX,yref:rightY,x0:rightRange[0],y0:rightRange[0],x1:rightRange[1],y1:rightRange[1],line:{{color:'#98A2B3',width:1,dash:'dot'}}}});
   annotations.push({{xref:'paper',yref:'paper',x:0.2275,y:yDomain[1]+0.0025,text:`${{ck}} · 2D P${{projectionA}}/P${{projectionB}}`,showarrow:false,font:{{size:12,color:'#1d232d'}}}},{{xref:'paper',yref:'paper',x:0.7725,y:yDomain[1]+0.0025,text:`${{ck}} · return P${{projection}}`,showarrow:false,font:{{size:12,color:'#1d232d'}}}});
 }});
 const plot=document.getElementById('plot'); plot.style.height=`${{layout.height}}px`; Plotly.react(plot,traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
}}
let timer; function schedule(){{clearTimeout(timer);timer=setTimeout(render,80);}}
document.getElementById('projection').addEventListener('change',render); document.getElementById('projectionPair').addEventListener('change',render); tokenBox.addEventListener('change',render); document.getElementById('startStep').addEventListener('input',schedule); document.getElementById('endStep').addEventListener('input',schedule); updateRangeLabels(); render();
</script></body></html>"""
    OUT_HTML.write_text(html.replace("__PLOTLY__", PLOTLY.read_text(encoding="utf-8")), encoding="utf-8")


def validate(series):
    checkpoints = {d["checkpoint"] for d in series}
    expected_rows = len(series) * LAST_TRANSITIONS
    actual_rows = sum(len(d["x_t"]) for d in series)
    assert actual_rows == expected_rows
    assert all(len(d["x_t"]) == LAST_TRANSITIONS for d in series)
    assert all(np.array_equal(d["t_plus_1"], d["t"] + 1) for d in series)
    assert all(np.allclose(d["x_t_plus_1"][:-1], d["x_t"][1:], rtol=0, atol=0) for d in series)
    return len(checkpoints), len(series), actual_rows


def main():
    series = load_series()
    n_checkpoints, n_series, n_rows = validate(series)
    write_csv(series)
    by_checkpoint, checkpoints = write_figures(series)
    write_html(by_checkpoint, checkpoints)
    print(json.dumps({
        "checkpoints": n_checkpoints,
        "series": n_series,
        "transitions_per_series": LAST_TRANSITIONS,
        "csv_rows": n_rows,
        "csv": str(OUT_CSV),
        "html": str(OUT_HTML),
        "grid": str(OUT_GRID),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
