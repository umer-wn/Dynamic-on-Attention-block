#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_io import atomic_json, atomic_jsonl, read_jsonl

from src.model_utils import load_model_and_tokenizer
from src.single_token_dynamics import SingleTokenOperator, exact_target_jacobian


MODEL = "EleutherAI/pythia-70m"
CACHE = "/home/luohaoming/model_feature_cache/hf_cache"
ROOT = "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan"
TOKENS = (
    "/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/"
    "frequency_audit/selected_tokens.jsonl"
)


def summary(matrix: torch.Tensor) -> dict:
    hidden = int(matrix.shape[0])
    return {
        "shape": list(matrix.shape),
        "frobenius_norm": float(matrix.norm().cpu()),
        "normalized_frobenius": float(matrix.norm().cpu()) / math.sqrt(hidden),
        "checksum_sha256": hashlib.sha256(matrix.detach().cpu().numpy().tobytes()).hexdigest(),
    }


def load_model(revision: str):
    model, tokenizer, device = load_model_and_tokenizer(
        MODEL, revision, MODEL, "float32", "cuda", CACHE, "eager", True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--reference-revision", default="step1000")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--frequency-manifest", default=TOKENS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--prepare-bank-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    token_rows = read_jsonl(Path(args.frequency_manifest))
    if len(token_rows) != 16:
        raise RuntimeError(f"expected 16 tokens, got {len(token_rows)}")
    bank_path = root / "manifests" / f"common_state_bank__{args.reference_revision}.pt"
    bank_meta_path = root / "manifests" / f"common_state_bank__{args.reference_revision}.json"
    if args.prepare_bank_only:
        model, _, device = load_model(args.reference_revision)
        ids = torch.tensor([int(row["token_id"]) for row in token_rows], device=device)
        with torch.no_grad():
            states = model.get_input_embeddings()(ids).float().cpu()
        torch.save({"token_ids": ids.cpu(), "states": states}, bank_path)
        metadata = {
            "reference_revision": args.reference_revision,
            "token_ids": ids.cpu().tolist(),
            "shape": list(states.shape),
            "bank_path": str(bank_path),
            "tensor_sha256": hashlib.sha256(states.numpy().tobytes()).hexdigest(),
        }
        atomic_json(bank_meta_path, metadata)
        print(json.dumps(metadata))
        return
    if not args.revision:
        raise ValueError("--revision is required unless --prepare-bank-only is used")
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    bank = torch.load(bank_path, map_location="cpu")
    token_ids = [int(row["token_id"]) for row in token_rows]
    if bank["token_ids"].tolist() != token_ids:
        raise RuntimeError("common state bank token ids do not match frequency manifest")

    started = time.perf_counter()
    model, _, device = load_model(args.revision)
    operator = SingleTokenOperator(model, "isolated_token")
    matrix_dir = root / "jacobians" / args.revision
    matrix_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for token_index, token_row in enumerate(token_rows):
        token_id = int(token_row["token_id"])
        with torch.no_grad():
            self_state = model.get_input_embeddings()(torch.tensor([token_id], device=device)).float()[0]
        common_state = bank["states"][token_index].to(device).float()
        conditions = [("self_t0", self_state)]
        if args.revision == args.reference_revision:
            conditions.append(("common_step1000_state", self_state))
        else:
            conditions.append(("common_step1000_state", common_state))
        computed: dict[str, tuple[torch.Tensor, Path]] = {}
        for condition, state in conditions:
            if condition == "common_step1000_state" and args.revision == args.reference_revision:
                matrix, matrix_path = computed["self_t0"]
            else:
                matrix = exact_target_jacobian(operator, state, chunk_size=args.chunk_size)
                matrix_path = matrix_dir / f"token{token_id}__{condition}.pt"
                torch.save(matrix.cpu(), matrix_path)
                computed[condition] = (matrix, matrix_path)
            rows.append(
                {
                    "checkpoint": args.revision,
                    "training_step": int(args.revision.removeprefix("step")),
                    "token_index": token_index,
                    "token_id": token_id,
                    "decoded": token_row["decoded"],
                    "frequency_bin": int(token_row["frequency_bin"]),
                    "condition": condition,
                    "reference_revision": args.reference_revision if condition.startswith("common") else None,
                    "matrix_path": str(matrix_path),
                    "jacobian_definition": "d final_hidden[0,0,:] / d inputs_embeds[0,0,:]",
                    **summary(matrix),
                }
            )
        print(json.dumps({"checkpoint": args.revision, "token": token_id,
                          "self": rows[-2]["normalized_frobenius"], "common": rows[-1]["normalized_frobenius"]}), flush=True)
    output = root / "raw" / args.revision / "jacobians_controls.jsonl"
    atomic_jsonl(output, rows)
    completion = {
        "status": "complete",
        "checkpoint": args.revision,
        "reference_revision": args.reference_revision,
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "chunk_size": args.chunk_size,
        "bank_metadata": str(bank_meta_path),
    }
    atomic_json(root / "raw" / args.revision / "controls_complete.json", completion)
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
