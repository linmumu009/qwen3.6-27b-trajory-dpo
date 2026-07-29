#!/usr/bin/env python3
"""Print metadata-only diagnostics for one shared flattened LoRA payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument(
        "--metadata-limit",
        type=int,
        default=0,
        help="Print the first N metadata records; zero prints only the summary.",
    )
    parser.add_argument(
        "--peft-adapter",
        type=Path,
        help="Also print PEFT safetensor key/shape metadata for comparison.",
    )
    args = parser.parse_args()

    path = args.payload.resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadatas = payload.get("metadatas")
    flattened_tensor = payload.get("flattened_tensor")
    if not isinstance(metadatas, list) or not metadatas:
        raise TypeError("payload metadatas must be a non-empty list")
    if not isinstance(flattened_tensor, torch.Tensor):
        raise TypeError("payload flattened_tensor must be a torch.Tensor")

    summary = {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "format": payload.get("format"),
        "peft_config": payload.get("peft_config"),
        "tensor_count": len(metadatas),
        "flattened_numel": flattened_tensor.numel(),
        "flattened_dtype": str(flattened_tensor.dtype),
        "flattened_finite": bool(torch.isfinite(flattened_tensor).all()),
        "flattened_nonzero": int(torch.count_nonzero(flattened_tensor)),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    for index, metadata in enumerate(metadatas[: args.metadata_limit]):
        print(
            json.dumps(
                {"index": index, **dict(metadata)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.peft_adapter:
        adapter_path = args.peft_adapter.resolve(strict=True)
        with safe_open(adapter_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            print(
                json.dumps(
                    {
                        "peft_adapter": str(adapter_path),
                        "peft_tensor_count": len(keys),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            for index, name in enumerate(keys[: args.metadata_limit]):
                tensor_slice = handle.get_slice(name)
                print(
                    json.dumps(
                        {
                            "peft_index": index,
                            "name": name,
                            "shape": tensor_slice.get_shape(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
