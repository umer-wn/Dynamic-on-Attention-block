#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts._bootstrap import require_packages
require_packages(["numpy", "torch", "transformers"])

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
EXP15 = REPO / "experiments_ordered/15_window_jacobian_token_projection/scripts/run_experiment15.py"
spec = importlib.util.spec_from_file_location("exp15base", EXP15)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

CHECKPOINTS = ["step27000", "step28000", "step39000", "step40000", "step58000", "step59000"]
TOKEN_MANIFEST = REPO / "experiments_ordered/18_fine_grained_window_jacobian/manifests/frequency_stratified_tokens_8.csv"
CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
STATE_ROOT = Path("/data1/luohaoming/model_feature_experiments/experiment23_fine_8token_states")


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True); temp=path.with_suffix(path.suffix+".tmp")
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with temp.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def tokens():
    return [{"selection_index":int(r["selection_index"]),"token_id":int(r["token_id"]),"token":r["token"],
             "wikitext_train_count":int(r["wikitext_train_count"]),"frequency_bin":int(r["frequency_bin"])} for r in read(TOKEN_MANIFEST)]


def states(checkpoint: str, toks: list[dict], device: torch.device):
    path=STATE_ROOT/"states"/f"{checkpoint}_states.pt"
    if path.exists(): return torch.load(path,map_location="cpu",weights_only=True)["states"].float()
    model=base.load_model(checkpoint,CACHE,device); ids=torch.tensor([r["token_id"] for r in toks],device=device)
    state=model.get_input_embeddings()(ids).detach().float(); values=[state.cpu()]
    mask=torch.ones((len(toks),1),device=device,dtype=torch.long); pos=torch.zeros((1,1),device=device,dtype=torch.long)
    with torch.inference_mode():
        for _ in range(1024):
            state=model.gpt_neox(inputs_embeds=state.unsqueeze(1),attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True).last_hidden_state[:,0,:].float()
            values.append(state.cpu())
    result=torch.stack(values);path.parent.mkdir(parents=True,exist_ok=True);torch.save({"checkpoint":checkpoint,"steps":1024,"token_ids":[r["token_id"] for r in toks],"states":result},path)
    del model;torch.cuda.empty_cache();return result


def refresh(toks: list[dict]):
    parts=ROOT/"processed/fine_metric_parts_8token"; rows=[]
    for cp in CHECKPOINTS:
        path=parts/f"{cp}.csv"
        if path.exists(): rows.extend(read(path))
    if not rows:return
    write(ROOT/"processed/jacobian_fine_grained_8tokens_raw.csv",rows)
    loss_rows=read(ROOT/"processed/fine_checkpoint_loss.csv") if (ROOT/"processed/fine_checkpoint_loss.csv").exists() else []
    losses={r["checkpoint"]:float(r["proof_pile2_test_loss"]) for r in loss_rows}
    out=[]
    for cp in CHECKPOINTS:
        for ds in range(0,1025,64):
            group=[r for r in rows if r["checkpoint"]==cp and int(r["dynamic_step"])==ds]
            if not group:continue
            rho=np.array([float(r["spectral_radius"]) for r in group]); fro=np.array([float(r["normalized_frobenius_norm"]) for r in group])
            out.append({"checkpoint":cp,"training_step":int(cp[4:]),"dynamic_step":ds,"token_count":len(group),
                        "spectral_radius_median":np.median(rho),"spectral_radius_min":rho.min(),"spectral_radius_max":rho.max(),
                        "normalized_frobenius_norm_median":np.median(fro),"normalized_frobenius_norm_min":fro.min(),"normalized_frobenius_norm_max":fro.max(),
                        "proof_pile2_test_loss":losses.get(cp,"")})
    write(ROOT/"processed/jacobian_fine_grained_8tokens.csv",out)


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:1");p.add_argument("--chunk",type=int,default=128);p.add_argument("--checkpoints",nargs="+",choices=CHECKPOINTS);args=p.parse_args();device=torch.device(args.device);toks=tokens();parts=ROOT/"processed/fine_metric_parts_8token";parts.mkdir(parents=True,exist_ok=True)
    expected=17*len(toks)
    for cp in (args.checkpoints or CHECKPOINTS):
        part=parts/f"{cp}.csv"; rows=read(part) if part.exists() else [];done={(int(r["dynamic_step"]),int(r["token_id"])) for r in rows}
        if len(done)==expected:print(json.dumps({"checkpoint":cp,"status":"skip"}),flush=True);continue
        trajectory=states(cp,toks,device);model=base.load_model(cp,CACHE,device);model.set_attn_implementation("eager")
        for ds in range(0,1025,64):
            for i,tok in enumerate(toks):
                if (ds,tok["token_id"]) in done:continue
                started=time.perf_counter();J=base.exact_jacobian(model,trajectory[ds,i].to(device),args.chunk)
                row={"checkpoint":cp,"training_step":int(cp[4:]),"dynamic_step":ds,**tok,
                     "spectral_radius":float(torch.linalg.eigvals(J).abs().max()),
                     "normalized_frobenius_norm":float(torch.linalg.vector_norm(J))/math.sqrt(J.shape[0]),
                     "runtime_seconds":time.perf_counter()-started}
                rows.append(row);done.add((ds,tok["token_id"]));write(part,rows);print(json.dumps({"stage":"metric","checkpoint":cp,"dynamic_step":ds,"token":tok["token"],"seconds":row["runtime_seconds"]},ensure_ascii=False),flush=True)
        del model,trajectory;torch.cuda.empty_cache()

if __name__=="__main__":main()
