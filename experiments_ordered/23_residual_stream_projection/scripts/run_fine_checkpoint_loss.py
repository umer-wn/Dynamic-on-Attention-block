#!/usr/bin/env python
from __future__ import annotations
import csv, importlib.util, json, math, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts._bootstrap import require_packages
require_packages(["torch","transformers"])
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];REPO=Path(__file__).resolve().parents[3]
SOURCE=REPO/"experiments_ordered/18_fine_grained_window_jacobian/scripts/compute_proof_pile2_loss.py"
spec=importlib.util.spec_from_file_location("lossbase",SOURCE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CHECKPOINTS=["step27000","step28000","step39000","step40000","step58000","step59000"]

def read(path):
 with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
 import argparse;p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:1");args=p.parse_args();device=torch.device(args.device)
 output=ROOT/"processed/fine_checkpoint_loss.csv";results=read(output) if output.exists() else [];done={r["checkpoint"] for r in results}
 manifest=Path("/home/luohaoming/proof_pile2/manifests/proof_pile2_test_512_seed1818.jsonl");rows=base.load_manifest(manifest)
 tokenizer=AutoTokenizer.from_pretrained(base.MODEL_NAME,revision="step0",cache_dir="/home/luohaoming/model_feature_cache/hf_cache",local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
 for cp in CHECKPOINTS:
  if cp in done:continue
  started=time.perf_counter();model=base.load_model(cp,Path("/home/luohaoming/model_feature_cache/hf_cache"),device);loss,n=base.evaluate_loss(model,rows,int(tokenizer.pad_token_id),device,16)
  result={"checkpoint":cp,"training_step":int(cp[4:]),"dataset":base.DATASET_NAME,"sample_seed":1818,"sample_count":len(rows),"sequence_length":64,"manifest_path":str(manifest),"proof_pile2_test_loss":loss,"proof_pile2_test_perplexity":math.exp(min(loss,20)),"proof_pile2_test_predicted_tokens":n,"runtime_seconds":time.perf_counter()-started}
  results.append(result);base.atomic_csv(output,results);print(json.dumps(result),flush=True);del model;torch.cuda.empty_cache()
if __name__=="__main__":main()
