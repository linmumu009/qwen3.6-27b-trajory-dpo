#!/usr/bin/env python3
"""Load one immutable shared-file LoRA payload into a running Swift rollout server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:28220")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    payload_path = args.payload.resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    actual_sha256 = file_sha256(payload_path)
    if args.expected_sha256 and actual_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"LoRA SHA256 mismatch: expected {args.expected_sha256}, got {actual_sha256}"
        )

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "llin-ms-swift-flat-lora-v1":
        raise RuntimeError(f"unexpected LoRA payload format: {payload.get('format')!r}")
    peft_config = payload.get("peft_config")
    metadatas = payload.get("metadatas")
    flattened_tensor = payload.get("flattened_tensor")
    if not isinstance(peft_config, dict):
        raise TypeError("payload peft_config must be an object")
    if not isinstance(metadatas, list) or not metadatas:
        raise TypeError("payload metadatas must be a non-empty list")
    if not isinstance(flattened_tensor, torch.Tensor):
        raise TypeError("payload flattened_tensor must be a torch.Tensor")

    response = requests.post(
        f"{args.base_url.rstrip('/')}/update_adapter_flattened_param/",
        json={"peft_config": peft_config, "metadatas": metadatas},
        timeout=args.timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"adapter update failed: HTTP {response.status_code}: {response.text}")
    body = response.json()
    if not body.get("all_workers_loaded", False):
        raise RuntimeError(f"rollout server did not confirm all workers: {body}")

    summary = {
        "payload": str(payload_path),
        "sha256": actual_sha256,
        "bytes": payload_path.stat().st_size,
        "tensor_count": len(metadatas),
        "flattened_numel": flattened_tensor.numel(),
        "flattened_dtype": str(flattened_tensor.dtype),
        "all_workers_loaded": True,
        "worker_result_count": len(body.get("dp_results") or []),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
