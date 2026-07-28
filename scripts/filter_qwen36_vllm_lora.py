#!/usr/bin/env python3
"""Build a vLLM-Ascend-compatible copy of a Qwen3.6 PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


UNSUPPORTED_MODULES = ("in_proj_a", "in_proj_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)

    source_weights = source / "adapter_model.safetensors"
    tensors = {}
    dropped = []
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            module_name = key.rsplit(".", 3)[-3]
            if module_name in UNSUPPORTED_MODULES:
                dropped.append(key)
            else:
                tensors[key] = handle.get_tensor(key)

    output_weights = destination / "adapter_model.safetensors"
    save_file(tensors, output_weights, metadata=metadata)

    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    target_modules = config.get("target_modules")
    if isinstance(target_modules, str):
        for module in UNSUPPORTED_MODULES:
            target_modules = target_modules.replace(f"|{module}", "")
        config["target_modules"] = target_modules
    config["exclude_modules"] = list(UNSUPPORTED_MODULES)
    (destination / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for name in ("additional_config.json", "args.json"):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)

    manifest = {
        "format": "qwen36-vllm-ascend-filtered-lora-v1",
        "source": str(source),
        "unsupported_modules_removed": list(UNSUPPORTED_MODULES),
        "source_tensor_count": len(tensors) + len(dropped),
        "kept_tensor_count": len(tensors),
        "dropped_tensor_count": len(dropped),
        "source_weights_sha256": sha256(source_weights),
        "output_weights_sha256": sha256(output_weights),
    }
    (destination / "compatibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
