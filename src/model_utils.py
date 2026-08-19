from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_and_tokenizer(
    model_name: str,
    revision: str,
    tokenizer_name: Optional[str],
    dtype: str,
    device: Optional[str],
    cache_dir: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    local_files_only: bool = False,
    tokenizer_revision: Optional[str] = None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install transformers and torch from requirements.txt") from exc

    torch_dtype = DTYPES.get(dtype, torch.float32)
    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        revision=(
            tokenizer_revision
            if tokenizer_revision is not None
            else (revision if tokenizer_name in (None, model_name) else None)
        ),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    model.to(dev)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer, dev


def get_transformer_blocks(model: torch.nn.Module) -> Union[torch.nn.ModuleList, list[torch.nn.Module]]:
    candidates = [
        "gpt_neox.layers",
        "model.layers",
        "transformer.h",
        "transformer.blocks",
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise ValueError("Could not locate transformer block list for this model architecture")


def expand_layer_selection(selection: Any, num_layers: int) -> list[int]:
    if selection == "all":
        return list(range(num_layers))
    if isinstance(selection, list):
        out: list[int] = []
        for item in selection:
            if item == "first":
                out.append(0)
            elif item == "middle":
                out.append(num_layers // 2)
            elif item == "last":
                out.append(num_layers - 1)
            else:
                idx = int(item)
                if idx < 0:
                    idx += num_layers
                out.append(idx)
        return sorted(set(i for i in out if 0 <= i < num_layers))
    return [int(selection)]


def iter_model_revisions(model_cfg: dict[str, Any]) -> list[str]:
    if "revisions" in model_cfg:
        return [str(x) for x in model_cfg["revisions"]]
    return [str(model_cfg.get("revision", "main"))]


@dataclass
class CapturedBlockCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _detach_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_detach_tree(v) for v in x)
    if isinstance(x, list):
        return [_detach_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _detach_tree(v) for k, v in x.items()}
    return x


def capture_block_call(model: torch.nn.Module, block: torch.nn.Module, batch: dict[str, torch.Tensor]) -> CapturedBlockCall:
    captured: dict[str, Any] = {}

    def hook(module, args, kwargs):
        captured["args"] = _detach_tree(args)
        captured["kwargs"] = _detach_tree(kwargs)

    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    finally:
        handle.remove()
    if "args" not in captured:
        raise RuntimeError("Forward hook did not capture block inputs")
    return CapturedBlockCall(args=captured["args"], kwargs=captured["kwargs"])


def cuda_memory() -> Optional[dict[str, int]]:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
    }
