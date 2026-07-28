#!/usr/bin/env python3
"""Summarize an online GRPO smoke run without printing trajectory contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            for child_key, numbers in numeric_leaves(item, path).items():
                result.setdefault(child_key, []).extend(numbers)
    elif isinstance(value, list):
        for item in value:
            for child_key, numbers in numeric_leaves(item, prefix).items():
                result.setdefault(child_key, []).extend(numbers)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result.setdefault(prefix, []).append(float(value))
    return result


def describe(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value)}
    return {"type": type(value).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_output", type=Path)
    args = parser.parse_args()
    output = args.run_output.resolve()
    completions_path = output / "completions.jsonl"
    logging_path = output / "logging.jsonl"
    checkpoint = output / "checkpoint-1"

    records = [
        json.loads(line)
        for line in completions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logs = [
        json.loads(line)
        for line in logging_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    selected_numeric: dict[str, list[float]] = {}
    for record in records:
        for key, values in numeric_leaves(record).items():
            if any(
                token in key.lower()
                for token in ("reward", "length", "score", "orm", "advantage")
            ):
                selected_numeric.setdefault(key, []).extend(values)

    numeric_summary = {}
    for key, values in sorted(selected_numeric.items()):
        numeric_summary[key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "all_finite": all(math.isfinite(value) for value in values),
        }

    adapter_path = checkpoint / "adapter_model.safetensors"
    checkpoint_summary: dict[str, Any] = {
        "exists": checkpoint.is_dir(),
        "latest_iteration": (
            checkpoint / "latest_checkpointed_iteration.txt"
        ).read_text(encoding="utf-8").strip(),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": sha256(adapter_path),
    }
    try:
        from safetensors import safe_open

        with safe_open(adapter_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            checkpoint_summary["adapter_tensor_count"] = len(keys)
            checkpoint_summary["first_adapter_keys"] = keys[:5]
            checkpoint_summary["all_adapter_tensors_finite"] = all(
                bool(handle.get_tensor(key).isfinite().all()) for key in keys
            )
    except ImportError:
        checkpoint_summary["safetensors_validation"] = "not_installed"

    summary = {
        "format": "llin-online-grpo-smoke-summary-v1",
        "output_dir": str(output),
        "completion_record_count": len(records),
        "completion_top_level": (
            {key: describe(value) for key, value in sorted(records[0].items())}
            if records
            else {}
        ),
        "selected_numeric": numeric_summary,
        "logging_records": logs,
        "completions_bytes": completions_path.stat().st_size,
        "completions_sha256": sha256(completions_path),
        "checkpoint": checkpoint_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
