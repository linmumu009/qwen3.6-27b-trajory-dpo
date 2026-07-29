#!/usr/bin/env python3
"""Freeze executable PI eval prompts in the same schema as online GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_eval_prompts", type=Path)
    parser.add_argument("verifier_manifest", type=Path)
    parser.add_argument("output_dataset", type=Path)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()

    source_path = args.source_eval_prompts.resolve(strict=True)
    verifier_path = args.verifier_manifest.resolve(strict=True)
    output_path = args.output_dataset.resolve()
    manifest_path = (
        args.output_manifest.resolve()
        if args.output_manifest
        else output_path.with_suffix(".manifest.json")
    )

    source_rows = read_jsonl(source_path)
    verifier_rows = read_jsonl(verifier_path)
    verifiers: dict[str, dict[str, Any]] = {}
    for row in verifier_rows:
        for key in ("verifier_id", "task_id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                verifiers[value] = row

    train_prompt_hashes: set[str] = set()
    train_path = None
    if args.train_dataset:
        train_path = args.train_dataset.resolve(strict=True)
        for row in read_jsonl(train_path):
            metadata = row.get("metadata") or {}
            prompt_hash = metadata.get("prompt_sha256")
            if not isinstance(prompt_hash, str) or not prompt_hash:
                raise ValueError("train row is missing metadata.prompt_sha256")
            train_prompt_hashes.add(prompt_hash)

    output_rows: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    environment_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for index, row in enumerate(source_rows):
        prompt = row.get("prompt")
        tools = row.get("tools")
        metadata = row.get("metadata")
        if not isinstance(prompt, list) or not prompt:
            raise TypeError(f"source row {index}: prompt must be a non-empty list")
        if not isinstance(tools, list):
            raise TypeError(f"source row {index}: tools must be a list")
        if not isinstance(metadata, dict):
            raise TypeError(f"source row {index}: metadata must be an object")
        environment = metadata.get("environment")
        if not isinstance(environment, dict):
            raise TypeError(f"source row {index}: metadata.environment must be an object")
        environment_id = environment.get("environment_id")
        task_id = environment.get("task_id")
        prompt_hash = metadata.get("prompt_sha256")
        if not isinstance(environment_id, str) or not environment_id.startswith("sft/"):
            raise ValueError(f"source row {index}: invalid environment_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"source row {index}: invalid task_id")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            raise ValueError(f"source row {index}: invalid prompt_sha256")
        verifier_id = f"{environment_id}:{task_id}"
        verifier = verifiers.get(verifier_id) or verifiers.get(task_id)
        if verifier is None:
            raise KeyError(f"source row {index}: no verifier for {verifier_id}")
        if verifier.get("environment_id") != environment_id:
            raise ValueError(f"source row {index}: verifier environment mismatch")
        if prompt_hash in prompt_hashes:
            raise ValueError(f"source row {index}: duplicate eval prompt hash")

        prompt_hashes.add(prompt_hash)
        task_ids.add(task_id)
        environment_ids.add(environment_id)
        output_rows.append(
            {
                "messages": prompt,
                "tools": tools,
                "metadata": {
                    **metadata,
                    "online_eval_source_index": index,
                    "verifier_id": verifier_id,
                },
                "verifier_id": verifier_id,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )

    overlap = sorted(prompt_hashes & train_prompt_hashes)
    if overlap:
        raise RuntimeError(f"train/eval prompt overlap: {overlap[:8]}")

    write_text_atomic(
        output_path,
        "".join(stable_json(row) + "\n" for row in output_rows),
    )
    manifest = {
        "format": "llin-online-grpo-eval-freeze-v1",
        "source_eval_prompts": str(source_path),
        "source_eval_prompts_sha256": file_sha256(source_path),
        "verifier_manifest": str(verifier_path),
        "verifier_manifest_sha256": file_sha256(verifier_path),
        "train_dataset": str(train_path) if train_path else None,
        "train_dataset_sha256": file_sha256(train_path) if train_path else None,
        "output_dataset": str(output_path),
        "output_dataset_sha256": file_sha256(output_path),
        "rows": len(output_rows),
        "unique_task_ids": len(task_ids),
        "unique_environment_ids": len(environment_ids),
        "unique_prompt_hashes": len(prompt_hashes),
        "train_prompt_overlap": 0,
        "claim_scope": "internal train-disjoint pilot; not a fresh external heldout",
    }
    write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
