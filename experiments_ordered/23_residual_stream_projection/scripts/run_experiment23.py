#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "yaml"])

import torch
import yaml
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiment23.yaml"
MAIN_FIELDS = [
    "checkpoint", "dynamic_step", "selection_index", "token_id", "token",
    "wikitext_train_count", "frequency_bin", "projection_1", "projection_2",
    "projection_3", "projection_4", "vector_kind", "vector_l2", "state_l2",
    "relative_update_l2", "projection_seed", "projection_sha256",
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_tokens(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [{
        "selection_index": int(row["selection_index"]),
        "token_id": int(row["token_id"]),
        "token": row["token"],
        "wikitext_train_count": int(row["wikitext_train_count"]),
        "frequency_bin": int(row["frequency_bin"]),
    } for row in rows]
    expected = [(21825, " clones"), (23778, " motive"), (19211, " cabinet"), (6574, " miles")]
    actual = [(row["token_id"], row["token"]) for row in result]
    if actual != expected:
        raise RuntimeError(f"Experiment 23 requires the exact Experiment 16 four-token manifest; got {actual}")
    return result


def generated_basis(hidden: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((hidden, 4), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.float()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_basis(config: dict) -> tuple[torch.Tensor, str]:
    saved = torch.load(config["experiment16_basis"], map_location="cpu", weights_only=True)
    basis = saved["basis"].float()
    regenerated = generated_basis(512, int(config["projection_seed"]))
    if int(saved["seed"]) != int(config["projection_seed"]):
        raise RuntimeError("Experiment 16 projection seed does not match Experiment 23 config")
    if not torch.equal(basis, regenerated):
        error = float((basis - regenerated).abs().max())
        raise RuntimeError(f"Experiment 16 projection basis mismatch: max_abs={error}")
    return basis, tensor_sha256(basis)


def load_model(config: dict, checkpoint: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=checkpoint, cache_dir=config["cache_dir"],
        local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True,
    ).to(device=device, dtype=torch.float32)
    model.eval()
    model.set_attn_implementation("eager")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class ResidualCapture:
    def __init__(self, model, capture_layers: bool = False):
        self.model = model
        self.capture_layers = capture_layers
        self.final_input: torch.Tensor | None = None
        self.layer_inputs: list[torch.Tensor] = []
        self.layer_outputs: list[torch.Tensor] = []
        self.handles = []

    def __enter__(self):
        def final_pre(_module, args):
            self.final_input = args[0][:, 0, :]

        self.handles.append(self.model.gpt_neox.final_layer_norm.register_forward_pre_hook(final_pre))
        if self.capture_layers:
            for layer in self.model.gpt_neox.layers:
                self.handles.append(layer.register_forward_pre_hook(self._layer_pre))
                self.handles.append(layer.register_forward_hook(self._layer_post))
        return self

    def _layer_pre(self, _module, args):
        self.layer_inputs.append(args[0][:, 0, :])

    def _layer_post(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.layer_outputs.append(hidden[:, 0, :])

    def reset(self):
        self.final_input = None
        self.layer_inputs.clear()
        self.layer_outputs.clear()

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()


def one_step(model, state: torch.Tensor, capture: ResidualCapture):
    capture.reset()
    batch = len(state)
    output = model.gpt_neox(
        inputs_embeds=state.unsqueeze(1),
        attention_mask=torch.ones((batch, 1), device=state.device, dtype=torch.long),
        position_ids=torch.zeros((1, 1), device=state.device, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, 0, :].float()
    if capture.final_input is None:
        raise RuntimeError("final LayerNorm input hook did not fire")
    pre_ln = capture.final_input.float()
    internal = pre_ln - state
    effective = output - state
    correction = output - pre_ln
    return output, pre_ln, internal, effective, correction


def atomic_csv(path: Path, rows: Iterable[dict], fields: list[str] = MAIN_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def vector_row(checkpoint: str, step: int, token: dict, vector_kind: str,
               vector: torch.Tensor, state: torch.Tensor, basis: torch.Tensor,
               seed: int, checksum: str) -> dict:
    coords = vector.detach().float().cpu() @ basis.T
    vector_norm = float(torch.linalg.vector_norm(vector.float()))
    state_norm = float(torch.linalg.vector_norm(state.float()))
    return {
        "checkpoint": checkpoint,
        "dynamic_step": step,
        **token,
        **{f"projection_{index + 1}": float(coords[index]) for index in range(4)},
        "vector_kind": vector_kind,
        "vector_l2": vector_norm,
        "state_l2": state_norm,
        "relative_update_l2": vector_norm / max(state_norm, 1e-12),
        "projection_seed": seed,
        "projection_sha256": checksum,
    }


def smoke(config: dict, tokens: list[dict], basis: torch.Tensor, checksum: str,
          device: torch.device) -> dict:
    settings = config["smoke"]
    token = next(row for row in tokens if row["token_id"] == int(settings["token_id"]))
    model = load_model(config, settings["checkpoint"], device)
    ids = torch.tensor([token["token_id"]], device=device)
    state = model.get_input_embeddings()(ids).detach().float()
    max_layer_error = 0.0
    max_identity_error = 0.0
    max_forward_error = 0.0
    with torch.inference_mode(), ResidualCapture(model, capture_layers=True) as capture:
        for _step in range(int(settings["steps"])):
            original = model.gpt_neox(
                inputs_embeds=state.unsqueeze(1),
                attention_mask=torch.ones((1, 1), device=device, dtype=torch.long),
                position_ids=torch.zeros((1, 1), device=device, dtype=torch.long),
                use_cache=False, return_dict=True,
            ).last_hidden_state[:, 0, :].float()
            output, pre_ln, internal, effective, correction = one_step(model, state, capture)
            layer_sum = sum((right - left) for left, right in zip(capture.layer_inputs, capture.layer_outputs))
            max_layer_error = max(max_layer_error, float((pre_ln - state - layer_sum).abs().max()))
            max_identity_error = max(max_identity_error, float((effective - internal - correction).abs().max()))
            max_forward_error = max(max_forward_error, float((output - original).abs().max()))
            state = output
    tolerance = float(settings["tolerance"])
    result = {
        "checkpoint": settings["checkpoint"], "token_id": token["token_id"],
        "token": token["token"], "steps": int(settings["steps"]),
        "max_layer_residual_identity_error": max_layer_error,
        "max_component_identity_error": max_identity_error,
        "max_hook_vs_original_forward_error": max_forward_error,
        "tolerance": tolerance, "projection_seed": int(config["projection_seed"]),
        "projection_sha256": checksum,
        "passed": max(max_layer_error, max_identity_error, max_forward_error) < tolerance,
    }
    atomic_json(ROOT / "processed/smoke_test.json", result)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    if not result["passed"]:
        raise RuntimeError(f"smoke test failed: {result}")
    print(json.dumps({"stage": "smoke", **result}, ensure_ascii=False), flush=True)
    return result


def run_checkpoint(config: dict, checkpoint: str, tokens: list[dict], basis: torch.Tensor,
                   checksum: str, device: torch.device, overwrite: bool):
    parts = ROOT / "processed/checkpoint_parts"
    main_path = parts / f"{checkpoint}_residual.csv"
    components_path = parts / f"{checkpoint}_components.csv"
    complete_path = parts / f"{checkpoint}_complete.json"
    if complete_path.exists() and main_path.exists() and components_path.exists() and not overwrite:
        print(json.dumps({"stage": "formal", "checkpoint": checkpoint, "status": "skip"}), flush=True)
        return
    started = time.perf_counter()
    model = load_model(config, checkpoint, device)
    ids = torch.tensor([row["token_id"] for row in tokens], device=device)
    state = model.get_input_embeddings()(ids).detach().float()
    main_rows, component_rows = [], []
    with torch.inference_mode(), ResidualCapture(model) as capture:
        for step in range(int(config["steps"])):
            next_state, _pre_ln, internal, effective, correction = one_step(model, state, capture)
            vectors = (("residual_internal", internal), ("effective_increment", effective),
                       ("final_ln_correction", correction))
            for token_index, token in enumerate(tokens):
                main_rows.append(vector_row(checkpoint, step, token, "residual_internal",
                                            internal[token_index], state[token_index], basis,
                                            int(config["projection_seed"]), checksum))
                for kind, values in vectors:
                    component_rows.append(vector_row(checkpoint, step, token, kind,
                                                     values[token_index], state[token_index], basis,
                                                     int(config["projection_seed"]), checksum))
            state = next_state
    atomic_csv(main_path, main_rows)
    atomic_csv(components_path, component_rows)
    metadata = {"checkpoint": checkpoint, "steps": int(config["steps"]),
                "token_ids": [row["token_id"] for row in tokens],
                "main_rows": len(main_rows), "component_rows": len(component_rows),
                "projection_seed": int(config["projection_seed"]),
                "projection_sha256": checksum, "runtime_seconds": time.perf_counter() - started,
                "status": "complete"}
    atomic_json(complete_path, metadata)
    print(json.dumps({"stage": "formal", **metadata}), flush=True)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def merge_csv(paths: list[Path], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    total = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MAIN_FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row); total += 1
    temporary.replace(output)
    return total


def formal(config: dict, tokens: list[dict], basis: torch.Tensor, checksum: str,
           device: torch.device, overwrite: bool, checkpoints: list[str] | None = None,
           output_stem: str = ""):
    checkpoints = checkpoints or list(config["checkpoints"])
    for checkpoint in checkpoints:
        run_checkpoint(config, checkpoint, tokens, basis, checksum, device, overwrite)
    parts = ROOT / "processed/checkpoint_parts"
    main = [parts / f"{checkpoint}_residual.csv" for checkpoint in checkpoints]
    components = [parts / f"{checkpoint}_components.csv" for checkpoint in checkpoints]
    suffix = f"_{output_stem}" if output_stem else ""
    main_count = merge_csv(main, ROOT / f"processed/residual_projection_trajectory{suffix}.csv")
    component_count = merge_csv(components, ROOT / f"processed/residual_projection_components{suffix}.csv")
    expected = len(checkpoints) * len(tokens) * int(config["steps"])
    if main_count != expected or component_count != 3 * expected:
        raise RuntimeError(f"row count mismatch: main={main_count}, components={component_count}, expected={expected}")
    atomic_json(ROOT / "processed/run_metadata.json", {
        "model": config["model"], "checkpoints": checkpoints,
        "steps": int(config["steps"]), "tokens": tokens, "main_rows": main_count,
        "component_rows": component_count, "projection_seed": int(config["projection_seed"]),
        "projection_sha256": checksum,
        "dynamic_step_semantics": "row t is the update x_t -> x_(t+1)",
        "primary_vector": "residual_internal = input_to_final_layer_norm - x_t",
    })
    print(json.dumps({"stage": "merge", "main_rows": main_count,
                      "component_rows": component_count}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint-set", choices=("base", "fine", "combined"), default="base")
    args = parser.parse_args()
    config = load_config(args.config)
    tokens = read_tokens(Path(config["token_manifest"]))
    basis, checksum = load_basis(config)
    device = torch.device(args.device)
    if args.stage in ("smoke", "all"):
        smoke(config, tokens, basis, checksum, device)
    if args.stage in ("formal", "all"):
        checkpoint_sets = {
            "base": list(config["checkpoints"]),
            "fine": list(config["fine_checkpoints"]),
            "combined": sorted(set(config["checkpoints"] + config["fine_checkpoints"]), key=lambda x: int(x[4:])),
        }
        chosen = checkpoint_sets[args.checkpoint_set]
        stem = "fine" if args.checkpoint_set == "fine" else "combined" if args.checkpoint_set == "combined" else ""
        formal(config, tokens, basis, checksum, device, args.overwrite, chosen, stem)


if __name__ == "__main__":
    main()
