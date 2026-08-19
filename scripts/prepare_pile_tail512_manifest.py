#!/usr/bin/env python3
import argparse, hashlib, json, os, random, time, traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def text_of(ex):
    for k in ("text", "content", "document"):
        v = ex.get(k)
        if isinstance(v, str):
            return v
    return None

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="EleutherAI/pile")
    ap.add_argument("--split-expr", default="test[-1000:]")
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--source-id", default="the_pile_test_tail512_random")
    ap.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()

    root = Path(args.root)
    manifest = root / "manifests" / f"{args.source_id}.jsonl"
    meta = root / "manifests" / f"{args.source_id}.metadata.json"
    status = root / "status" / f"{args.source_id}_prepare_status.json"
    sources_path = root / "manifests" / "sources.json"
    t0 = time.time()
    atomic_json(status, {
        "status": "running", "dataset": args.dataset, "split_expr": args.split_expr,
        "sample_count": args.sample_count, "seed": args.seed, "source_id": args.source_id,
        "endpoint": os.environ.get("HF_ENDPOINT"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    })
    try:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split_expr, trust_remote_code=True, num_proc=args.num_proc)
        n = len(ds)
        idxs = list(range(n))
        random.Random(args.seed).shuffle(idxs)
        idxs = idxs[:min(args.sample_count, n)]
        rows = []
        for j, i in enumerate(idxs):
            ex = ds[int(i)]
            text = text_of(ex)
            if not text or len(text.strip()) < 16:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows.append({
                "source_id": args.source_id,
                "dataset_name": args.dataset,
                "config": None,
                "split": "test",
                "split_expr": args.split_expr,
                "sample_index": len(rows),
                "original_index_in_tail_window": int(i),
                "text": text,
                "text_sha256": h,
            })
        if not rows:
            raise RuntimeError("no valid text rows sampled")
        with manifest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metadata = {
            "source_id": args.source_id,
            "dataset": args.dataset,
            "split": "test",
            "split_expr": args.split_expr,
            "tail_window": n,
            "sample_count_requested": args.sample_count,
            "sample_count": len(rows),
            "seed": args.seed,
            "manifest_path": str(manifest),
            "metadata_path": str(meta),
            "text_sha256_manifest": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "elapsed_seconds": time.time() - t0,
            "endpoint": os.environ.get("HF_ENDPOINT"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(meta, metadata)
        if sources_path.exists():
            sources = json.loads(sources_path.read_text(encoding="utf-8"))
        else:
            sources = {}
        sources[args.source_id] = {
            "dataset_name": args.dataset,
            "split": "test",
            "manifest_path": str(manifest),
            "metadata_path": str(meta),
            "sample_count": len(rows),
            "description": "Tail-window random 512 sample from The Pile test via datasets split expression.",
        }
        atomic_json(sources_path, sources)
        atomic_json(status, {**metadata, "status": "complete"})
        print(json.dumps({"status":"complete", "sample_count": len(rows), "tail_window": n, "manifest_path": str(manifest)}, ensure_ascii=False))
    except Exception as e:
        err = {"status":"failed", "error": repr(e), "traceback": traceback.format_exc(), "elapsed_seconds": time.time()-t0}
        atomic_json(status, err)
        print(json.dumps(err, ensure_ascii=False))
        raise

if __name__ == "__main__":
    main()
