#!/usr/bin/env python3
"""Build a self-contained Experiment 17/22/23 interactive dashboard."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments_ordered/17_visualize"
STATE_TRAJECTORY = ROOT / "experiments_ordered/16_frequency_stratified_window_jacobian/processed/projection_trajectory.csv"
STATE_TRAJECTORY_COMBINED = ROOT / "experiments_ordered/23_residual_stream_projection/processed/state_projection_trajectory_combined.csv"
METRICS = ROOT / "experiments_ordered/18_fine_grained_window_jacobian/processed/jacobian_fine_grained_8tokens.csv"
FINE_METRICS = ROOT / "experiments_ordered/23_residual_stream_projection/processed/jacobian_fine_grained_8tokens.csv"
ENDPOINT_100 = ROOT / "experiments_ordered/20_100token_endpoint_jacobian/processed/checkpoint_metric_summary.csv"
PROJECTED_JACOBIAN = ROOT / "experiments_ordered/20_100token_endpoint_jacobian/processed/projected_jacobian_checkpoint_summary.csv"
BASE_LOSS = ROOT / "experiments_ordered/18_fine_grained_window_jacobian/processed/proof_pile2_test_loss_by_checkpoint.csv"
FINE_LOSS = ROOT / "experiments_ordered/23_residual_stream_projection/processed/fine_checkpoint_loss.csv"
PERTURB_TRAJ = ROOT / "experiments_ordered/22_multiscale_perturbation_stability/processed/perturbation_projection_trajectories.csv"
PERTURB_ENDPOINTS = ROOT / "experiments_ordered/22_multiscale_perturbation_stability/processed/perturbation_projection_endpoints.csv"
PERTURB_TRAJ_FALLBACK = EXP / "processed/single_token_perturbation_projection_trajectories.csv"
PERTURB_RESULTS = ROOT / "experiments_ordered/22_multiscale_perturbation_stability/processed/perturbation_outcomes.csv"
PLOTLY = EXP / "plotly-2.35.2.min.js"
OUTPUT = EXP / "dynamic_step_projection_visualization.html"
DENSE = ROOT / "experiments_ordered/25_dense_checkpoint_suite/processed"
DENSE_STATE = DENSE / "state_projection_trajectory.csv"
DENSE_RESIDUAL = DENSE / "residual_projection_trajectory.csv"
DENSE_METRICS = DENSE / "jacobian8_summary.csv"
DENSE_ENDPOINT = DENSE / "endpoint100_summary.csv"
DENSE_LOSS = DENSE / "checkpoint_loss.csv"
DENSE_PERTURB_ENDPOINTS = DENSE / "perturbation_projection_endpoints.csv"
DENSE_PERTURB_OUTCOMES = DENSE / "perturbation_outcomes.csv"
DENSE_SCREEN = DENSE / "period_screen_summary.csv"
DENSE_CONVERGENCE = DENSE / "convergence_detail.csv"
HLE_LOSS = ROOT / "experiments_ordered/24_hle_subset_loss/processed/hle_loss_verified_n116_all59.csv"
EXP19_SYSTEM = ROOT / "experiments_ordered/19_single_token_attractor_validation/processed/stage4_system_summary.csv"

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return v

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def read_filtered(path, predicate):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if predicate(row)]

def trajectories(rows):
    grouped={}
    for r in rows:
        key=(r["checkpoint"],int(r["selection_index"]))
        s=grouped.setdefault(key,{"id":int(r["selection_index"]),"token":r["token"],"token_id":int(r["token_id"]),"step":[],"p1":[],"p2":[],"p3":[],"p4":[]})
        s["step"].append(int(r["dynamic_step"]))
        for d in range(1,5): s[f"p{d}"].append(round(float(r[f"projection_{d}"]),7))
    cps={}
    for (cp,_),s in grouped.items():
        order=sorted(range(len(s["step"])),key=s["step"].__getitem__)
        for k in ("step","p1","p2","p3","p4"): s[k]=[s[k][i] for i in order]
        cps.setdefault(cp,[]).append(s)
    return [{"checkpoint":cp,"training_step":int(cp[4:]),"series":sorted(ss,key=lambda x:x["id"])} for cp,ss in sorted(cps.items(),key=lambda x:int(x[0][4:]))]

def perturb_trajectories(rows):
    if rows and "original_prev_projection_1" in rows[0]:
        result=[]
        for r in rows:
            epsilon=float(r.get("epsilon") or r.get("relative_scale"))
            s={"checkpoint":r["checkpoint"],"token":r.get("token", ""),"token_id":int(r["token_id"]),"injection_step":int(r.get("injection_step") or 0),"epsilon":epsilon,"direction_id":int(r.get("direction_id") or 0),"step":[1023,1024]}
            for d in range(1,5):
                s[f"o{d}"]=[round(float(r[f"original_prev_projection_{d}"]),7),round(float(r[f"original_final_projection_{d}"]),7)]
                s[f"q{d}"]=[round(float(r[f"perturbed_prev_projection_{d}"]),7),round(float(r[f"perturbed_final_projection_{d}"]),7)]
            result.append(s)
        return result
    grouped={}
    for r in rows:
        epsilon=float(r.get("epsilon") or r.get("relative_scale")); seed=int(r.get("perturbation_seed") or 0)
        key=(r["checkpoint"],int(r["token_id"]),epsilon,int(r.get("injection_step") or 0),int(r.get("direction_id") or 0),seed)
        s=grouped.setdefault(key,{"checkpoint":r["checkpoint"],"token":r["token"],"token_id":int(r["token_id"]),"injection_step":int(r.get("injection_step") or 0),"epsilon":epsilon,"direction_id":int(r.get("direction_id") or 0),"step":[],"o1":[],"o2":[],"o3":[],"o4":[],"q1":[],"q2":[],"q3":[],"q4":[]})
        s["step"].append(int(r["dynamic_step"]))
        for d in range(1,5):
            s[f"q{d}"].append(round(float(r[f"perturbed_projection_{d}"]),7))
            s[f"o{d}"].append(round(float(r.get(f"original_projection_{d}") or r[f"perturbed_projection_{d}"]),7))
    for s in grouped.values():
        order=sorted(range(len(s["step"])),key=s["step"].__getitem__)
        for k in ("step","o1","o2","o3","o4","q1","q2","q3","q4"): s[k]=[s[k][i] for i in order]
    return list(grouped.values())

def write_perturb_endpoints(rows):
    paths=perturb_trajectories(rows)
    fields=["checkpoint","token_id","token","injection_step","epsilon","direction_id"]
    for prefix in ("original_prev","original_final","perturbed_prev","perturbed_final"):
        fields.extend(f"{prefix}_projection_{d}" for d in range(1,5))
    with PERTURB_ENDPOINTS.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for p in paths:
            if 1023 not in p["step"] or 1024 not in p["step"]: continue
            prev,final=p["step"].index(1023),p["step"].index(1024)
            row={k:p[k] for k in fields[:6]}
            for d in range(1,5):
                row[f"original_prev_projection_{d}"]=p[f"o{d}"][prev]
                row[f"original_final_projection_{d}"]=p[f"o{d}"][final]
                row[f"perturbed_prev_projection_{d}"]=p[f"q{d}"][prev]
                row[f"perturbed_final_projection_{d}"]=p[f"q{d}"][final]
            writer.writerow(row)
    return paths

def payload():
    dense = DENSE_STATE.exists() and DENSE_METRICS.exists()
    if dense:
        early_metrics=[r for r in read(METRICS) if 0<int(r["training_step"])<=10000]
        metric_source=[*early_metrics,*read(DENSE_METRICS)]
        early_losses=[r for r in read(BASE_LOSS) if 0<int(r["training_step"])<=10000] if BASE_LOSS.exists() else []
        loss_source=[*early_losses,*read(DENSE_LOSS)] if DENSE_LOSS.exists() else early_losses
    else:
        metric_source=[*read(METRICS), *(read(FINE_METRICS) if FINE_METRICS.exists() else [])]
        loss_source=[*(read(BASE_LOSS) if BASE_LOSS.exists() else []), *(read(FINE_LOSS) if FINE_LOSS.exists() else [])]
    losses = {row["checkpoint"]: num(row.get("proof_pile2_test_loss", "")) for row in loss_source}
    hle_losses = {row["checkpoint"]: num(row.get("hle_answer_token_loss", "")) for row in read(HLE_LOSS)} if HLE_LOSS.exists() else {}
    for row in metric_source:
        if not row.get("normalized_frobenius_norm_median") and row.get("jacobian_frobenius_norm_median"):
            row["normalized_frobenius_norm_median"] = float(row["jacobian_frobenius_norm_median"]) / (512 ** 0.5)
            row["normalized_frobenius_norm_min"] = float(row["jacobian_frobenius_norm_min"]) / (512 ** 0.5)
            row["normalized_frobenius_norm_max"] = float(row["jacobian_frobenius_norm_max"]) / (512 ** 0.5)
        if not row.get("proof_pile2_test_loss") and row["checkpoint"] in losses:
            row["proof_pile2_test_loss"] = losses[row["checkpoint"]]
        if row["checkpoint"] in hle_losses:
            row["hle_answer_token_loss"] = hle_losses[row["checkpoint"]]
    metric_by_key = {(r["checkpoint"], r["dynamic_step"]): r for r in metric_source}
    metric_rows=[{k:num(v) for k,v in r.items()} for r in metric_by_key.values()]
    endpoint_rows=[]
    projected_by_cp={r["checkpoint"]:{k:num(v) for k,v in r.items()} for r in read(PROJECTED_JACOBIAN)} if PROJECTED_JACOBIAN.exists() else {}
    if dense and DENSE_ENDPOINT.exists():
        endpoint_source=[r for r in read(ENDPOINT_100) if 0<int(r["training_step"])<=10000]
        endpoint_source.extend(read(DENSE_ENDPOINT))
    else:
        endpoint_source=read(ENDPOINT_100) if ENDPOINT_100.exists() else []
    screen_by_cp={}
    if dense and DENSE_SCREEN.exists():
        grouped={}
        for row in read(DENSE_SCREEN): grouped.setdefault(row["checkpoint"],[]).append(row)
        for cp,rows in grouped.items():
            screen_by_cp[cp]={"fixed_point_error_median":statistics.median(float(r["fixed_point_error"]) for r in rows),"best_period_error_median":statistics.median(float(r["best_period_error"]) for r in rows),"best_period_median":statistics.median(float(r["best_period"]) for r in rows)}
    # Period screening is checkpoint-level rather than dynamic-step-level.
    # Repeat the checkpoint value on every available dynamic-step slice so it
    # can be selected in the main feature view and compared on the second Y axis.
    for row in metric_rows:
        row.update(screen_by_cp.get(row["checkpoint"], {}))
    convergence_by_cp={r["checkpoint"]:r for r in read(DENSE_CONVERGENCE)} if dense and DENSE_CONVERGENCE.exists() else {}
    if endpoint_source:
        for row in endpoint_source:
            converted={k:num(v) for k,v in row.items()}
            converted.update({k:v for k,v in projected_by_cp.get(row["checkpoint"],{}).items() if k not in {"checkpoint","training_step","dynamic_step","token_count","token_selection","token_seed"}})
            converted["proof_pile2_test_loss"]=losses.get(row["checkpoint"], "")
            converted.update(screen_by_cp.get(row["checkpoint"],{}))
            if row["checkpoint"] in convergence_by_cp:
                detail=convergence_by_cp[row["checkpoint"]
                ]
                for key in ("period","regression_median","shooting_initial_median","shooting_final_median","shooting_reduction_ratio_median","shooting_status","floquet_estimate_median","floquet_lower","floquet_upper","floquet_status"):
                    converted[key]=num(detail.get(key,""))
            endpoint_rows.append(converted)
    if dense:
        early_state=[r for r in read(STATE_TRAJECTORY_COMBINED) if 0<int(r["checkpoint"][4:])<=10000] if STATE_TRAJECTORY_COMBINED.exists() else []
        state_source=[*early_state,*read(DENSE_STATE)]
    else:
        state_path=STATE_TRAJECTORY_COMBINED if STATE_TRAJECTORY_COMBINED.exists() else STATE_TRAJECTORY
        state_source=read(state_path)
    if dense and DENSE_PERTURB_ENDPOINTS.exists():
        # The perturbation protocol uses its own fixed 100-token sample, which
        # is intentionally distinct from the four state-trajectory tokens.
        # Embed a deterministic four-token preview; uploading the CSV exposes
        # all 100 perturbation tokens without making the built-in HTML huge.
        perturb_preview_ids=[]
        with DENSE_PERTURB_ENDPOINTS.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                token_id=int(row["token_id"])
                if int(row.get("direction_id") or 0)==0 and token_id not in perturb_preview_ids:
                    perturb_preview_ids.append(token_id)
                    if len(perturb_preview_ids)==4: break
        perturb_preview_ids=set(perturb_preview_ids)
        perturb_source=read_filtered(DENSE_PERTURB_ENDPOINTS,lambda r:int(r["token_id"]) in perturb_preview_ids and int(r.get("direction_id") or 0)==0)
        outcome_rows=[]
    else:
        outcome_rows=[{k:num(v) for k,v in r.items()} for r in read(PERTURB_RESULTS)] if PERTURB_RESULTS.exists() else []
        if PERTURB_TRAJ.exists(): write_perturb_endpoints(read(PERTURB_TRAJ))
        perturb_path=PERTURB_ENDPOINTS if PERTURB_ENDPOINTS.exists() else (PERTURB_TRAJ if PERTURB_TRAJ.exists() else PERTURB_TRAJ_FALLBACK)
        perturb_source=read(perturb_path) if perturb_path.exists() else []
    if dense and DENSE_RESIDUAL.exists():
        early_residual=[r for r in read(ROOT / "experiments_ordered/23_residual_stream_projection/processed/residual_projection_trajectory_combined.csv") if 0<int(r["checkpoint"][4:])<=10000]
        residual_source=[*early_residual,*read(DENSE_RESIDUAL)]
    else:
        residual_source=[]
    exp19_rows=[]
    if EXP19_SYSTEM.exists():
        grouped={}
        for row in read(EXP19_SYSTEM): grouped.setdefault(row["checkpoint"],[]).append(row)
        for cp,rows in sorted(grouped.items(),key=lambda item:int(item[0][4:])):
            labels={label:sum(r["stability"]==label for r in rows) for label in ("stable","boundary","unstable")}
            values=sorted(float(r["leading_multiplier_modulus"]) for r in rows)
            residuals=sorted(float(r["shooting_normalized_residual_p95"]) for r in rows)
            exp19_rows.append({"checkpoint":cp,"training_step":int(cp[4:]),"systems":len(rows),"periods":sorted({int(r["minimal_period"]) for r in rows}),"shooting_residual_median":statistics.median(residuals),"rho_median":statistics.median(values),**labels,"attractor_status":"not_validated"})
    return {"stateTrajectories":trajectories(state_source),"residualTrajectories":trajectories(residual_source),"metrics":metric_rows,"endpointMetrics100":endpoint_rows,"experiment19Summary":exp19_rows,"perturbTrajectories":perturb_trajectories(perturb_source),"perturbOutcomes":outcome_rows}

def build(data):
    page=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dynamic-step projections</title><script>__PLOTLY__</script><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#202734;font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1540px;margin:auto;padding:18px}h1{font-size:20px;margin:0}.sub,.muted,.status{color:#647084}.sub{margin-top:5px}.row,.controls,.checks{display:flex;flex-wrap:wrap;gap:9px;align-items:center}.uploads,.controls{margin:14px 0}.upload,.control,.panel,.scope{background:#fff;border:1px solid #dfe4ec;border-radius:9px;padding:10px}.upload{flex:1;min-width:270px}.upload label,.title{font-weight:650;color:#344054}.upload input{display:block;width:100%;margin-top:7px}.status{font-size:11px;margin-top:6px}.control{min-width:155px}.title{margin-bottom:7px}.checks{max-width:560px}.checks label{background:#f1f5f9;border-radius:4px;padding:3px 5px}select,input[type=number]{border:1px solid #ccd4df;border-radius:5px;padding:5px;background:#fff}input[type=range]{accent-color:#2878b5}.plots{display:grid;grid-template-columns:1fr 1fr;gap:14px}.head{display:flex;justify-content:space-between;align-items:baseline;gap:10px}.head h2{font-size:14px;margin:0}.plot{height:500px}.featureplot{height:430px}.panel.feature{margin-top:14px}.scope{margin-top:10px;font-variant-numeric:tabular-nums}.tablewrap{overflow-x:auto;margin-top:10px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;border-bottom:1px solid #e4e7ec;text-align:left;white-space:nowrap}th{background:#f8fafc;color:#344054;font-weight:650}.stable{color:#067647}.unstable{color:#b42318}.boundary,.borderline{color:#b54708}.error{color:#b42318}button{padding:5px 9px}@media(max-width:900px){.plots{grid-template-columns:1fr}.plot{height:430px}}
</style></head><body><main class="page"><h1>Dynamic-step state, residual, features and perturbations</h1><div class="sub">状态轨迹、残差 trace、雅可比/收敛特征与扰动终点使用独立变量；CSV 仅在浏览器本地解析。</div>
<section class="row uploads"><div class="upload"><label>状态轨迹 CSV<input id="stateFile" type="file" accept=".csv,text/csv"></label><div id="stateStatus" class="status">内置59个checkpoint：step2000–10000已有数据 + step11000–61000密集数据；排除step0</div></div><div class="upload"><label>残差 trace CSV<input id="residualFile" type="file" accept=".csv,text/csv"></label><div id="residualStatus" class="status">内置59个checkpoint残差流；排除step0</div></div><div class="upload"><label>8-token动态特征 CSV<input id="featureFile" type="file" accept=".csv,text/csv"></label><div id="featureStatus" class="status">内置早期+密集checkpoint的17 dynamic-step核心指标，并合并19点HLE loss</div></div><div class="upload"><label>100-token终点/收敛特征 CSV<input id="endpointFile" type="file" accept=".csv,text/csv"></label><div id="endpointStatus" class="status">内置59个checkpoint，并加入P1–P4低维投影 Jacobian 范数；可上传 endpoint100_with_projected_jacobian.csv</div></div><div class="upload"><label>扰动投影终点 CSV<input id="perturbFile" type="file" accept=".csv,text/csv"></label><div id="perturbStatus" class="status">扰动协议采用独立100-token样本；内置固定前4个token×direction0。完整数据请上传实验25 perturbation_projection_endpoints.csv，不能上传 perturbation_outcomes.csv</div></div><button id="resetData">恢复内置数据</button></section>
<section class="controls"><div class="control"><div class="title">Checkpoint <span id="ckLabel"></span></div><div class="row"><select id="ck"></select><span id="ckStep"></span></div><div id="ckCoverage" class="status"></div></div><div class="control"><div class="title">Dynamic step <span id="windowLabel"></span></div><div class="row"><input id="start" type="range"><input id="end" type="range"></div></div><div class="control"><div class="title">Token</div><div id="tokens" class="checks"></div></div><div class="control"><div class="title">轨迹维度</div><label><input id="threeD" type="checkbox"> 3D P1/P2/P3</label></div><div class="control"><div class="title">Return map</div><select id="returnDim"><option value="1">P1</option><option value="2">P2</option><option value="3">P3</option><option value="4">P4</option></select></div></section>
<section class="controls"><div class="control"><div class="title">终点显示</div><label><input id="showOriginalFinal" type="checkbox" checked> x_final</label> <label><input id="showPerturbedFinal" type="checkbox" checked> x̃_final</label></div><div class="control"><div class="title">注入步</div><select id="pinj"></select></div><div class="control"><div class="title">扰动大小 ε</div><select id="peps"></select></div><div class="control"><div class="title">扰动 token</div><select id="ptoken"></select></div><div class="control"><div class="title">方向</div><select id="pdir"></select></div></section>
<section class="plots"><div class="panel"><div class="head"><h2 id="leftTitle"></h2><span class="muted">浅→深表示 dynamic step 增大；仅状态图可叠加扰动终点</span></div><div id="left" class="plot"></div></div><div class="panel"><div class="head"><h2 id="rightTitle"></h2><span class="muted">状态 Return map</span></div><div id="right" class="plot"></div></div></section><div id="scope" class="scope"></div>
<section class="plots" style="margin-top:14px"><div class="panel"><div class="head"><h2 id="residualLeftTitle"></h2><span class="muted">Experiment 23: f(x_t)，不显示扰动数据</span></div><div id="residualLeft" class="plot"></div></div><div class="panel"><div class="head"><h2 id="residualRightTitle"></h2><span class="muted">残差更新自身的相邻动态步 Return map</span></div><div id="residualRight" class="plot"></div></div></section><div id="residualScope" class="scope"></div>
<section class="panel feature"><div class="head"><h2>核心特征视图</h2><span class="muted">最多同时选择 2 项；双特征使用左右独立 Y 轴</span></div><div class="controls"><label>指标协议 <select id="metricProtocol"><option value="8">8-token动态曲线（中位数）</option><option value="100">100-token终点（均值）</option></select></label><div><div class="title">特征（最多2项）</div><div id="featureChecks" class="checks"></div><div id="featureLimit" class="status"></div></div><div><div class="title">Dynamic step（多选）</div><div id="metricSteps" class="checks"></div></div><label>视图 <select id="featureView"><option value="2d">2D双Y轴曲线</option><option value="2.5d">2.5D（仅单特征）</option></select></label><label>X轴 <select id="fx"><option>linear</option><option>log</option></select></label><label>Y轴 <select id="fy"><option>linear</option><option>log</option></select></label></div><div id="featurePlot" class="featureplot"></div><div id="featureScope" class="scope"></div></section>
<section class="panel feature"><div class="head"><h2>当前 checkpoint 的动力学判定</h2><span class="muted">非核心曲线指标改为表格；不展示扰动恢复比例</span></div><div id="currentDynamicsTable" class="tablewrap"></div></section>
</main><script>
const embedded=__DATA__;let data=structuredClone(embedded);const $=id=>document.getElementById(id),colors=['#2878b5','#d95f02','#2a9d8f','#8e5bb7','#7a7f87','#bc5090'];
const metrics={spectral_radius:{label:'谱半径 ρ(J)',mid:'spectral_radius_median',mean:'spectral_radius_mean'},normalized_frobenius_norm:{label:'归一化 Frobenius ||J||F/√512',mid:'normalized_frobenius_norm_median',mean:'normalized_frobenius_norm_mean'},projected_normalized_frobenius_d1:{label:'低维 Jacobian ||J₁||F（P1）',mean:'projected_normalized_frobenius_d1_mean'},projected_normalized_frobenius_d2:{label:'低维 Jacobian ||J₂||F/√2（P1–P2）',mean:'projected_normalized_frobenius_d2_mean'},projected_normalized_frobenius_d3:{label:'低维 Jacobian ||J₃||F/√3（P1–P3）',mean:'projected_normalized_frobenius_d3_mean'},projected_normalized_frobenius_d4:{label:'低维 Jacobian ||J₄||F/2（P1–P4）',mean:'projected_normalized_frobenius_d4_mean'},lyapunov_exponent_last_256:{label:'Lyapunov指数（最后256步）',mean:'lyapunov_exponent_last_256_mean'},best_period:{label:'候选周期 p*（checkpoint中位数）',mid:'best_period_median',mean:'best_period_median'},proof_pile2_test_loss:{label:'Proof Pile 2 loss',mid:'proof_pile2_test_loss',mean:'proof_pile2_test_loss'},hle_answer_token_loss:{label:'HLE answer-token loss',mid:'hle_answer_token_loss',mean:'hle_answer_token_loss'}};
$('featureStatus').textContent='内置早期+密集checkpoint的17 dynamic-step核心指标，并合59点HLE loss';
function csv(text){text=text.replace(/^\uFEFF/,'');let rows=[],r=[],f='',q=false;for(let i=0;i<text.length;i++){let c=text[i];if(q){if(c==='"'&&text[i+1]==='"'){f+='"';i++}else if(c==='"')q=false;else f+=c}else if(c==='"')q=true;else if(c===','){r.push(f);f=''}else if(c==='\n'){r.push(f.replace(/\r$/,''));if(r.some(Boolean))rows.push(r);r=[];f=''}else f+=c}if(f||r.length){r.push(f);rows.push(r)}const h=rows.shift().map(x=>x.trim());return rows.map(v=>Object.fromEntries(h.map((x,i)=>[x,v[i]??''])))}
function required(rows,cols,name){if(!rows.length)throw Error(name+'为空');const miss=cols.filter(c=>!(c in rows[0]));if(miss.length)throw Error('缺少列: '+miss.join(', '))}
function trajRows(rows){required(rows,['checkpoint','dynamic_step','selection_index','token_id','token','projection_1','projection_2','projection_3','projection_4'],'轨迹');const g=new Map();for(const r of rows){const k=r.checkpoint+'|'+r.selection_index;if(!g.has(k))g.set(k,{checkpoint:r.checkpoint,id:+r.selection_index,token:r.token,token_id:+r.token_id,step:[],p1:[],p2:[],p3:[],p4:[]});const s=g.get(k);s.step.push(+r.dynamic_step);for(let d=1;d<=4;d++)s['p'+d].push(+r['projection_'+d])}const cps=new Map();for(const s of g.values()){const o=s.step.map((_,i)=>i).sort((a,b)=>s.step[a]-s.step[b]);for(const k of ['step','p1','p2','p3','p4'])s[k]=o.map(i=>s[k][i]);if(!cps.has(s.checkpoint))cps.set(s.checkpoint,[]);cps.get(s.checkpoint).push(s)}return [...cps].map(([checkpoint,series])=>({checkpoint,training_step:+checkpoint.replace('step',''),series})).sort((a,b)=>a.training_step-b.training_step)}
function metricRows(rows){required(rows,['checkpoint','training_step','dynamic_step','spectral_radius_median'],'特征');return rows.map(r=>{const o=Object.fromEntries(Object.entries(r).map(([k,v])=>[k,k==='checkpoint'?v:+v]));if(!Number.isFinite(o.normalized_frobenius_norm_median)&&Number.isFinite(o.jacobian_frobenius_norm_median))o.normalized_frobenius_norm_median=o.jacobian_frobenius_norm_median/Math.sqrt(512);return o})}
function perturbRows(rows){if(rows.length&&'endpoint_distance' in rows[0]&&!('perturbed_projection_1' in rows[0]))throw Error('perturbation_outcomes.csv 不含投影坐标，不能用于绘制终点；请上传 perturbation_projection_endpoints.csv');const endpoint=rows.length&&'original_prev_projection_1' in rows[0],g=new Map();required(rows,endpoint?['checkpoint','token_id','original_prev_projection_1','original_final_projection_1','perturbed_prev_projection_1','perturbed_final_projection_1']:['checkpoint','dynamic_step','token_id','perturbed_projection_1'],endpoint?'扰动投影终点':'扰动投影轨迹');for(const r of rows){const eps=+(r.epsilon||r.relative_scale),k=[r.checkpoint,r.token_id,eps,r.injection_step||0,r.direction_id||0,r.perturbation_seed||r.seed||0].join('|');if(!g.has(k))g.set(k,{checkpoint:r.checkpoint,token:r.token||'',token_id:+r.token_id,injection_step:+(r.injection_step||0),epsilon:eps,direction_id:+(r.direction_id||0),step:[],o1:[],o2:[],o3:[],o4:[],q1:[],q2:[],q3:[],q4:[]});const s=g.get(k);if(endpoint){s.step.push(1023,1024);for(let d=1;d<=4;d++){s['o'+d].push(+r['original_prev_projection_'+d],+r['original_final_projection_'+d]);s['q'+d].push(+r['perturbed_prev_projection_'+d],+r['perturbed_final_projection_'+d])}}else{s.step.push(+r.dynamic_step);for(let d=1;d<=4;d++){s['q'+d].push(+r['perturbed_projection_'+d]);s['o'+d].push(+(r['original_projection_'+d]||r['perturbed_projection_'+d]))}}}for(const s of g.values()){const o=s.step.map((_,i)=>i).sort((a,b)=>s.step[a]-s.step[b]);for(const k of ['step','o1','o2','o3','o4','q1','q2','q3','q4'])s[k]=o.map(i=>s[k][i])}data.perturbTrajectories=[...g.values()];return endpoint?'原始/扰动投影终点':'完整轨迹（仅使用 1023→1024 终点）'}
function rangeText(values){return values.length?`${unique(values).length}个[${Math.min(...values)}–${Math.max(...values)}]`:'0个'}
function checkpointSummary(){const state=data.stateTrajectories.map(r=>r.training_step),residual=data.residualTrajectories.map(r=>r.training_step),feature=data.metrics.map(r=>r.training_step),perturb=data.perturbTrajectories.map(r=>+r.checkpoint.replace('step',''));return `状态 ${rangeText(state)}；残差 ${rangeText(residual)}；特征 ${rangeText(feature)}；扰动 ${rangeText(perturb)}`}
async function load(input,status,kind){const file=input.files[0];if(!file)return;try{const rows=csv(await file.text());let detail='';if(kind==='state')data.stateTrajectories=trajRows(rows);else if(kind==='residual')data.residualTrajectories=trajRows(rows);else if(kind==='metric')data.metrics=metricRows(rows);else if(kind==='endpoint')data.endpointMetrics100=metricRows(rows);else detail=' · '+perturbRows(rows);status.textContent=`${file.name} · ${rows.length.toLocaleString()}行${detail} · ${checkpointSummary()}`;status.classList.remove('error');init()}catch(e){status.textContent=e.message;status.classList.add('error')}}
function unique(values){return [...new Set(values)].sort((a,b)=>a-b)}function options(el,values,all=false){el.innerHTML=(all?'<option value="all">全部</option>':'')+values.map(v=>`<option value="${v}">${v}</option>`).join('')}function rgba(h,a){const n=parseInt(h.slice(1),16);return `rgba(${n>>16},${n>>8&255},${n&255},${a})`}function checkpointName(){return $('ck').value}function findCheckpoint(rows){return rows.find(d=>d.checkpoint===checkpointName())}function checkpointNames(){return [...new Set([...data.stateTrajectories,...data.residualTrajectories,...data.perturbTrajectories].map(d=>d.checkpoint))].sort((a,b)=>+a.replace('step','')-+b.replace('step',''))}function checked(root){return new Set([...document.querySelectorAll('#'+root+' input:checked')].map(x=>x.value))}function near(a,v){let b=0;for(let i=1;i<a.length;i++)if(Math.abs(a[i]-v)<Math.abs(a[b]-v))b=i;return b}
function tokenControls(){const stateCheckpoint=findCheckpoint(data.stateTrajectories),residualCheckpoint=findCheckpoint(data.residualTrajectories),series=stateCheckpoint?.series||residualCheckpoint?.series||[];$('tokens').innerHTML=series.map((s,i)=>`<label><input type="checkbox" value="${s.id}" ${i<4?'checked':''}>${s.token}</label>`).join('')}function ranges(force=false){const stateCheckpoint=findCheckpoint(data.stateTrajectories),residualCheckpoint=findCheckpoint(data.residualTrajectories),steps=[...(stateCheckpoint?.series||[]),...(residualCheckpoint?.series||[])].flatMap(s=>s.step);if(!steps.length)return;const lo=Math.min(...steps),hi=Math.max(...steps);for(const e of [$('start'),$('end')]){e.min=lo;e.max=hi}if(force){$('start').value=Math.max(lo,hi-512);$('end').value=hi}}
function perturbControls(){const paths=data.perturbTrajectories,local=paths.filter(p=>p.checkpoint===checkpointName()),all=local.length?local:paths,pathDefault=local[0]||paths[0],keep={i:$('pinj').value,e:$('peps').value,t:$('ptoken').value,d:$('pdir').value};options($('pinj'),unique(all.map(r=>+r.injection_step)));options($('peps'),unique(all.map(r=>+r.epsilon)));options($('ptoken'),unique(all.map(r=>+r.token_id)),true);options($('pdir'),unique(all.map(r=>+r.direction_id)),true);if(pathDefault&&!keep.e){$('pinj').value=String(pathDefault.injection_step);$('peps').value=String(pathDefault.epsilon);$('ptoken').value=String(pathDefault.token_id);$('pdir').value=String(pathDefault.direction_id)}for(const [id,v] of Object.entries(keep)){const el={i:$('pinj'),e:$('peps'),t:$('ptoken'),d:$('pdir')}[id];if(v&&[...el.options].some(o=>o.value===v))el.value=v}}
function selectedPerturbPaths(){if(!$('showOriginalFinal').checked&&!$('showPerturbedFinal').checked)return[];return data.perturbTrajectories.filter(path=>path.checkpoint===checkpointName()&&+path.injection_step===+$('pinj').value&&+path.epsilon===+$('peps').value&&($('ptoken').value==='all'||+path.token_id===+$('ptoken').value)&&($('pdir').value==='all'||+path.direction_id===+$('pdir').value))}
function renderTrajectoryPanels(checkpointData,leftId,rightId,isState){const selectedIds=checked('tokens'),startStep=Math.min(+$('start').value,+$('end').value),endStep=Math.max(+$('start').value,+$('end').value),projection=+$('returnDim').value,is3d=$('threeD').checked,leftTraces=[],rightTraces=[],baseLayout={margin:{l:62,r:20,t:40,b:52},paper_bgcolor:'#fff',plot_bgcolor:'#fff',legend:{orientation:'h',y:1.1}};if(checkpointData)checkpointData.series.filter(series=>selectedIds.has(String(series.id))).forEach((series,j)=>{const lo=near(series.step,startStep),hi=near(series.step,endStep),indices=Array.from({length:Math.max(0,hi-lo+1)},(_,k)=>lo+k),color=colors[j%colors.length],shade=indices.map((_,k)=>rgba(color,.12+.88*k/Math.max(1,indices.length-1))),identity={name:`${series.token} · ${series.token_id}`};if(is3d)leftTraces.push({...identity,type:'scatter3d',mode:'lines+markers',x:indices.map(i=>series.p1[i]),y:indices.map(i=>series.p2[i]),z:indices.map(i=>series.p3[i]),marker:{size:3,color:shade},line:{color:rgba(color,.4)}});else leftTraces.push({...identity,type:'scatter',mode:'lines+markers',x:indices.map(i=>series.p1[i]),y:indices.map(i=>series.p2[i]),marker:{size:4,color:shade},line:{color:rgba(color,.4)}});const pairs=indices.slice(0,-1);rightTraces.push({...identity,type:'scatter',mode:'lines+markers',x:pairs.map(i=>series['p'+projection][i]),y:pairs.map(i=>series['p'+projection][i+1]),marker:{size:4,color:shade.slice(0,-1)},line:{color:rgba(color,.4)}})});if(isState)for(const perturbationPath of selectedPerturbPaths()){const finalIndex=perturbationPath.step.indexOf(1024),previousIndex=perturbationPath.step.indexOf(1023);if(finalIndex<0||previousIndex<0)continue;const showOriginal=$('showOriginalFinal').checked,showPerturbed=$('showPerturbedFinal').checked;if(showOriginal)leftTraces.push(is3d?{type:'scatter3d',mode:'markers+text',x:[perturbationPath.o1[finalIndex]],y:[perturbationPath.o2[finalIndex]],z:[perturbationPath.o3[finalIndex]],text:['x_final'],textposition:'top center',marker:{size:7,color:'#111827',symbol:'circle'},name:'x_final'}:{type:'scatter',mode:'markers+text',x:[perturbationPath.o1[finalIndex]],y:[perturbationPath.o2[finalIndex]],text:['x_final'],textposition:'top center',marker:{size:11,color:'#111827',symbol:'circle'},name:'x_final'});if(showPerturbed)leftTraces.push(is3d?{type:'scatter3d',mode:'markers+text',x:[perturbationPath.q1[finalIndex]],y:[perturbationPath.q2[finalIndex]],z:[perturbationPath.q3[finalIndex]],text:['x̃_final'],textposition:'top center',marker:{size:7,color:'#b42318',symbol:'diamond'},name:'x̃_final'}:{type:'scatter',mode:'markers+text',x:[perturbationPath.q1[finalIndex]],y:[perturbationPath.q2[finalIndex]],text:['x̃_final'],textposition:'top center',marker:{size:11,color:'#b42318',symbol:'diamond'},name:'x̃_final'});if(showOriginal)rightTraces.push({type:'scatter',mode:'markers+text',x:[perturbationPath['o'+projection][previousIndex]],y:[perturbationPath['o'+projection][finalIndex]],text:['x_final'],textposition:'top center',marker:{size:11,color:'#111827',symbol:'circle'},name:'x_final'});if(showPerturbed)rightTraces.push({type:'scatter',mode:'markers+text',x:[perturbationPath['q'+projection][previousIndex]],y:[perturbationPath['q'+projection][finalIndex]],text:['x̃_final'],textposition:'top center',marker:{size:11,color:'#b42318',symbol:'diamond'},name:'x̃_final'})}const emptyAnnotation=checkpointData?[]:[{text:isState?'该 checkpoint 无状态轨迹':'请上传 Experiment 23 residual trace CSV，或该 checkpoint 无残差轨迹',showarrow:false,font:{color:'#647084'}}],leftLayout=is3d?{...baseLayout,scene:{xaxis:{title:'P1'},yaxis:{title:'P2'},zaxis:{title:'P3'}},annotations:emptyAnnotation}:{...baseLayout,xaxis:{title:'P1'},yaxis:{title:'P2'},annotations:emptyAnnotation},symbol=isState?'x':'f';Plotly.react(leftId,leftTraces,leftLayout,{responsive:true,displaylogo:false});Plotly.react(rightId,rightTraces,{...baseLayout,xaxis:{title:`P${projection}: ${symbol}_t`},yaxis:{title:`P${projection}: ${symbol}_{t+1}`},annotations:emptyAnnotation},{responsive:true,displaylogo:false})}
function renderTop(){const stateCheckpoint=findCheckpoint(data.stateTrajectories),residualCheckpoint=findCheckpoint(data.residualTrajectories),startStep=Math.min(+$('start').value,+$('end').value),endStep=Math.max(+$('start').value,+$('end').value),projection=+$('returnDim').value,is3d=$('threeD').checked,selectedIds=[...checked('tokens')],trainingStep=+checkpointName().replace('step','');$('ckLabel').textContent=checkpointName();$('ckStep').textContent=trainingStep;$('windowLabel').textContent=`${startStep}–${endStep}`;$('leftTitle').textContent=is3d?'状态轨迹 P1/P2/P3':'状态轨迹 P1/P2';$('rightTitle').textContent=`状态 P${projection}: x_t → x_{t+1}`;$('residualLeftTitle').textContent=is3d?'残差更新轨迹 P1/P2/P3: f(x_t)':'残差更新轨迹 P1/P2: f(x_t)';$('residualRightTitle').textContent=`残差 P${projection}: f_t → f_{t+1}`;renderTrajectoryPanels(stateCheckpoint,'left','right',true);renderTrajectoryPanels(residualCheckpoint,'residualLeft','residualRight',false);$('scope').textContent=`状态图范围：checkpoint ${checkpointName()}（training step ${trainingStep}）；dynamic step ${startStep}–${endStep}；selection_index ${selectedIds.join(', ')}；扰动终点筛选：注入步 ${$('pinj').value}，ε=${$('peps').value}，token=${$('ptoken').value}，direction=${$('pdir').value}`;$('residualScope').textContent=`残差图范围：checkpoint ${checkpointName()}；dynamic step ${startStep}–${endStep}；selection_index ${selectedIds.join(', ')}；数据源与扰动终点完全隔离`}
function activeMetrics(){return $('metricProtocol').value==='100'?data.endpointMetrics100:data.metrics}function metricField(m){return $('metricProtocol').value==='100'?m.mean:m.mid}
function featureControls(){const source=activeMetrics(),oldF=[...checked('featureChecks')].slice(0,2),oldS=checked('metricSteps');$('featureChecks').innerHTML=Object.entries(metrics).filter(([k,m])=>metricField(m)&&source.some(r=>Number.isFinite(r[metricField(m)]))).map(([k,m],i)=>`<label><input type="checkbox" value="${k}" ${oldF.includes(k)||(!oldF.length&&i<2)?'checked':''}>${m.label}</label>`).join('');const steps=unique(source.map(r=>r.dynamic_step));$('metricSteps').innerHTML=steps.map(v=>`<label><input type="checkbox" value="${v}" ${oldS.has(String(v))||(!oldS.size&&v===steps.at(-1))?'checked':''}>${v}</label>`).join('');$('featureLimit').textContent='已选择 '+checked('featureChecks').size+'/2'}
function renderFeatures(){
 const source=activeMetrics(),fs=[...checked('featureChecks')].slice(0,2),steps=[...checked('metricSteps')].map(Number).sort((a,b)=>a-b),stepSet=new Set(steps),rows=source.filter(r=>stepSet.has(r.dynamic_step)),view=fs.length===2?'2d':$('featureView').value,tr=[],currentStep=+checkpointName().replace('step','');
 if(!rows.length||!fs.length){$('featureScope').textContent='特征图展示范围：当前协议、特征或 dynamic step 选择无可显示数据';Plotly.react('featurePlot',[],{annotations:[{text:'当前协议无可显示数据',showarrow:false}]},{responsive:true,displaylogo:false});return}
 const checkpoints=unique(rows.map(r=>r.training_step)),ckFirst=checkpoints[0],ckLast=checkpoints.at(-1),dsFirst=steps[0],dsLast=steps.at(-1),checkpointSpan=checkpoints.length===1?`step${ckFirst}`:`step${ckFirst}–step${ckLast}（${checkpoints.length} 个 checkpoint）`,dynamicSpan=steps.length===1?`${dsFirst}`:`${dsFirst}–${dsLast}（选中 ${steps.length} 个采样点：${steps.join(', ')}）`,plotRangeTitle=`checkpoint ${checkpointSpan} · dynamic step ${dynamicSpan}`;
 if(view==='2.5d'){
  const key=fs[0],m=metrics[key],field=metricField(m),xs=checkpoints,ys=steps,z=ys.map(ds=>xs.map(x=>{const r=rows.find(q=>q.dynamic_step===ds&&q.training_step===x);return r&&Number.isFinite(r[field])?r[field]:null}));tr.push({type:'surface',x:xs,y:ys,z,connectgaps:true,opacity:.9,name:m.label,showscale:true,contours:{z:{show:true,usecolormap:true,project:{z:true}}},colorscale:'Viridis'});for(const ds of ys){const rr=rows.filter(r=>r.dynamic_step===ds&&Number.isFinite(r[field])).sort((a,b)=>a.training_step-b.training_step);tr.push({type:'scatter3d',mode:'lines',x:rr.map(r=>r.training_step),y:rr.map(()=>ds),z:rr.map(r=>r[field]),line:{color:colors[0],width:3},name:`${m.label} · t=${ds}`,showlegend:false})}
  Plotly.react('featurePlot',tr,{title:{text:plotRangeTitle,font:{size:13}},margin:{l:20,r:20,t:55,b:20},scene:{xaxis:{title:'checkpoint / training step'},yaxis:{title:'dynamic step'},zaxis:{title:m.label}},paper_bgcolor:'#fff'},{responsive:true,displaylogo:false})
 }else{
  fs.forEach((key,j)=>{const m=metrics[key],field=metricField(m),axis=j===0?'y':'y2';for(const ds of steps){const rr=rows.filter(r=>r.dynamic_step===ds&&Number.isFinite(r[field])).sort((a,b)=>a.training_step-b.training_step);tr.push({type:'scatter',mode:'lines+markers',x:rr.map(r=>r.training_step),y:rr.map(r=>r[field]),yaxis:axis,name:`${m.label} · t=${ds}`,line:{color:colors[j],dash:['solid','dash','dot','dashdot'][steps.indexOf(ds)%4]},marker:{size:5,color:colors[j]}});const current=rr.find(r=>r.training_step===currentStep);if(current)tr.push({type:'scatter',mode:'markers',x:[currentStep],y:[current[field]],yaxis:axis,name:`当前 ${checkpointName()} · ${m.label}`,marker:{size:12,color:colors[j],symbol:j?'diamond-open':'circle-open',line:{width:3,color:colors[j]}},showlegend:false})}});
  const layout={title:{text:plotRangeTitle,font:{size:13}},margin:{l:72,r:fs.length===2?72:24,t:58,b:55},xaxis:{title:'checkpoint / training step',type:$('fx').value},yaxis:{title:metrics[fs[0]].label,type:$('fy').value,titlefont:{color:colors[0]},tickfont:{color:colors[0]}},legend:{orientation:'h'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',shapes:[{type:'line',xref:'x',yref:'paper',x0:currentStep,x1:currentStep,y0:0,y1:1,line:{color:'#111827',width:2,dash:'dot'}}],annotations:[{xref:'x',yref:'paper',x:currentStep,y:1,text:`当前 ${checkpointName()}`,showarrow:true,arrowhead:2,ax:0,ay:-28,bgcolor:'#fff'}]};if(fs.length===2)layout.yaxis2={title:metrics[fs[1]].label,type:$('fy').value,titlefont:{color:colors[1]},tickfont:{color:colors[1]},overlaying:'y',side:'right',showgrid:false};
  Plotly.react('featurePlot',tr,layout,{responsive:true,displaylogo:false})
 }
 const protocol=$('metricProtocol').value==='100'?'100-token均值':'8-token中位数';$('featureScope').textContent=`特征图展示范围：checkpoint ${checkpointSpan}；dynamic step ${dynamicSpan}；当前 checkpoint ${checkpointName()} 已用竖直虚线和空心标记高亮；指标协议 ${protocol}；特征 ${fs.map(k=>metrics[k].label).join(' / ')}`
}
function fnum(value,digits=4){return Number.isFinite(+value)?(+value).toExponential(Math.abs(+value)<.001?2:digits).replace('e+','e'):'—'}
function renderDynamicsTables(){const cp=checkpointName(),endpoint=data.endpointMetrics100.find(r=>r.checkpoint===cp),exact=data.experiment19Summary.find(r=>r.checkpoint===cp),parts=[];if(endpoint){const bestP=endpoint.best_period_median,fp=endpoint.fixed_point_error_median,bestErr=endpoint.best_period_error_median,rho=endpoint.floquet_estimate_median,shoot=endpoint.shooting_final_median,local=Number.isFinite(rho)?(rho<.98?'stable':rho>1.02?'unstable':'boundary'):'未做详细Floquet';parts.push(['固定点检验 p=1',fnum(fp),bestP===1?'固定点候选':'非最优周期','p=1仅用于区分不动点']);parts.push(['周期回归 p≤256',`p=${Math.round(bestP)}；误差 ${fnum(bestErr)}`,bestP>1?'周期候选':'固定点候选','候选不等于已验证吸引子']);parts.push(['多点射击',Number.isFinite(shoot)?`末端残差 ${fnum(shoot)}`:'—',endpoint.shooting_status||'未做详细评估','Experiment 19闭合阈值为p95≤1e-5；此处协议不同']);parts.push(['Floquet / monodromy',Number.isFinite(rho)?`ρ≈${fnum(rho)}；区间 [${fnum(endpoint.floquet_lower)}, ${fnum(endpoint.floquet_upper)}]`:'—',local,Number.isFinite(rho)?'按Experiment 19的ρ阈值映射；无16/32维一致性时仅作参考':'未入选详细评估'])}if(exact)parts.push(['Experiment 19原始结果',`系统 ${exact.systems}；周期 ${exact.periods.join('/')}；ρ中位数 ${fnum(exact.rho_median)}`,`stable ${exact.stable} / boundary ${exact.boundary} / unstable ${exact.unstable}`,'Stage 4结论：未验证稳定吸引周期轨道']);if(!parts.length)parts.push(['当前checkpoint','—','无动力学判定数据','早期checkpoint仅补入已有轨迹/核心特征']);$('currentDynamicsTable').innerHTML='<table><thead><tr><th>项目</th><th>数值</th><th>标签</th><th>解释</th></tr></thead><tbody>'+parts.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td class="${String(r[2]).split(' ')[0]}">${r[2]}</td><td>${r[3]}</td></tr>`).join('')+'</tbody></table>'}
function render(){renderTop();renderFeatures();renderDynamicsTables()}function init(){const checkpointSelect=$('ck'),previous=checkpointSelect.value,names=checkpointNames();checkpointSelect.innerHTML=names.map(name=>`<option value="${name}">${name} · training ${+name.replace('step','')}</option>`).join('');checkpointSelect.value=names.includes(previous)?previous:(names.includes('step57000')?'step57000':names.at(-1));$('ckCoverage').textContent=`状态 ${data.stateTrajectories.length} 个 checkpoint；残差 ${data.residualTrajectories.length} 个 checkpoint；可选并集：${names.join(', ')}`;tokenControls();ranges(true);perturbControls();featureControls();render()}
$('stateFile').onchange=()=>load($('stateFile'),$('stateStatus'),'state');$('residualFile').onchange=()=>load($('residualFile'),$('residualStatus'),'residual');$('featureFile').onchange=()=>load($('featureFile'),$('featureStatus'),'metric');$('endpointFile').onchange=()=>load($('endpointFile'),$('endpointStatus'),'endpoint');$('perturbFile').onchange=()=>load($('perturbFile'),$('perturbStatus'),'perturb');$('resetData').onclick=()=>{data=structuredClone(embedded);init()};$('ck').oninput=()=>{tokenControls();ranges(true);perturbControls();render()};$('tokens').onchange=render;$('featureChecks').onchange=e=>{const selected=[...document.querySelectorAll('#featureChecks input:checked')];if(selected.length>2){e.target.checked=false;$('featureLimit').textContent='最多同时选择2个特征'}else $('featureLimit').textContent='已选择 '+selected.length+'/2';if(checked('featureChecks').size===2)$('featureView').value='2d';renderFeatures()};$('metricSteps').onchange=renderFeatures;$('metricProtocol').onchange=()=>{$('featureView').value='2d';featureControls();renderFeatures()};$('featureView').onchange=()=>{if($('featureView').value==='2.5d'){const selected=[...document.querySelectorAll('#featureChecks input:checked')];selected.slice(1).forEach(x=>x.checked=false);$('featureLimit').textContent='2.5D仅保留第1个特征';document.querySelectorAll('#metricSteps input').forEach(x=>x.checked=true)}renderFeatures()};for(const id of ['start','end','threeD','returnDim','showOriginalFinal','showPerturbedFinal','pinj','peps','ptoken','pdir','fx','fy'])$(id).addEventListener('input',render);init();
</script></body></html>'''
    return page.replace("__PLOTLY__",PLOTLY.read_text(encoding="utf-8")).replace("__DATA__",json.dumps(data,ensure_ascii=False,separators=(",",":")))

def main():
    data=payload();OUTPUT.write_text(build(data),encoding="utf-8");print(json.dumps({"output":str(OUTPUT),"state_checkpoints":len(data["stateTrajectories"]),"residual_checkpoints":len(data["residualTrajectories"]),"metric_rows_8token":len(data["metrics"]),"metric_rows_100token":len(data["endpointMetrics100"]),"perturb_paths":len(data["perturbTrajectories"]),"perturb_outcomes":len(data["perturbOutcomes"])}))
if __name__=="__main__": main()






