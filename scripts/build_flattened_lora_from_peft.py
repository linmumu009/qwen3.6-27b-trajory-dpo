#!/usr/bin/env python3
"""Rebuild an MS-Swift shared flattened LoRA payload from a PEFT safetensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open


DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_payload", type=Path)
    parser.add_argument("peft_adapter", type=Path)
    parser.add_argument("output_payload", type=Path)
    parser.add_argument(
        "--expect-template-tensor-match",
        action="store_true",
        help="Fail unless rebuilt bytes exactly equal the template flattened tensor.",
    )
    args = parser.parse_args()

    template_path = args.template_payload.resolve(strict=True)
    adapter_path = args.peft_adapter.resolve(strict=True)
    output_path = args.output_payload.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(template_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "llin-ms-swift-flat-lora-v1":
        raise RuntimeError(f"unexpected template format: {payload.get('format')!r}")
    metadatas = payload.get("metadatas")
    template_tensor = payload.get("flattened_tensor")
    if not isinstance(metadatas, list) or not metadatas:
        raise TypeError("template metadatas must be a non-empty list")
    if not isinstance(template_tensor, torch.Tensor):
        raise TypeError("template flattened_tensor must be a torch.Tensor")
    if template_tensor.dtype is not torch.uint8:
        raise TypeError(
            f"template flattened tensor must use byte packing, got {template_tensor.dtype}"
        )

    expected_names = [str(item["name"]) for item in metadatas]
    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("template contains duplicate tensor names")

    packed_parts: list[torch.Tensor] = []
    source_nonzero = 0
    source_finite = True
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        expected_set = set(expected_names)
        if actual_names != expected_set:
            missing = sorted(expected_set - actual_names)
            extra = sorted(actual_names - expected_set)
            raise RuntimeError(
                f"PEFT/template key mismatch: missing={missing[:8]}, extra={extra[:8]}"
            )

        previous_end = 0
        expected_bytes = 0
        for metadata in metadatas:
            name = str(metadata["name"])
            shape = tuple(int(value) for value in metadata["shape"])
            numel = int(metadata["numel"])
            start_idx = int(metadata["start_idx"])
            end_idx = int(metadata["end_idx"])
            if start_idx != previous_end or end_idx - start_idx != numel:
                raise RuntimeError(f"non-contiguous template metadata at {name}")
            dtype_name = str(metadata["dtype"])
            try:
                dtype = DTYPES[dtype_name]
            except KeyError as exc:
                raise RuntimeError(f"unsupported template dtype {dtype_name} at {name}") from exc
            element_count = int(torch.tensor(shape).prod())
            expected_tensor_bytes = element_count * torch.empty((), dtype=dtype).element_size()
            if numel != expected_tensor_bytes:
                raise RuntimeError(
                    f"invalid template shape/byte-count at {name}: "
                    f"{numel} != {expected_tensor_bytes}"
                )

            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"shape mismatch at {name}: expected {shape}, got {tuple(tensor.shape)}"
                )
            tensor = tensor.to(dtype=dtype).contiguous()
            source_nonzero += int(torch.count_nonzero(tensor))
            source_finite = source_finite and bool(torch.isfinite(tensor).all())
            packed = tensor.view(torch.uint8).flatten()
            if packed.numel() != numel:
                raise RuntimeError(f"packed byte-count mismatch at {name}")
            packed_parts.append(packed)
            expected_bytes += packed.numel()
            previous_end = end_idx

    rebuilt_tensor = torch.cat(packed_parts)
    if rebuilt_tensor.numel() != expected_bytes:
        raise RuntimeError("rebuilt flattened byte count is internally inconsistent")
    if rebuilt_tensor.numel() != template_tensor.numel():
        raise RuntimeError(
            "rebuilt/template byte count mismatch: "
            f"{rebuilt_tensor.numel()} != {template_tensor.numel()}"
        )

    template_matches = bool(torch.equal(rebuilt_tensor, template_tensor))
    if args.expect_template_tensor_match and not template_matches:
        raise RuntimeError(
            "rebuilt PEFT bytes do not exactly match the template flattened tensor"
        )

    output = dict(payload)
    output.update(
        {
            "created_at": time.time(),
            "flattened_tensor": rebuilt_tensor,
            "source_peft_adapter": str(adapter_path),
            "source_peft_sha256": file_sha256(adapter_path),
            "template_payload": str(template_path),
            "template_payload_sha256": file_sha256(template_path),
        }
    )
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(output, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    summary = {
        "output_payload": str(output_path),
        "output_sha256": file_sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "tensor_count": len(metadatas),
        "flattened_numel": rebuilt_tensor.numel(),
        "flattened_sha256": tensor_sha256(rebuilt_tensor),
        "template_flattened_sha256": tensor_sha256(template_tensor),
        "template_tensor_matches": template_matches,
        "source_nonzero": source_nonzero,
        "source_finite": source_finite,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
