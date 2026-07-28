#!/usr/bin/env python3
"""Freeze and audit the deterministic P-001 train/heldout split.

The source preference JSONL and its manifest must be row-aligned.  P-001 uses
v15 and v20 for training and reserves v21 as an internal heldout set.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRAIN_VERSIONS = ("v15", "v20")
DEFAULT_HOLDOUT_VERSIONS = ("v21",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-versions", nargs="+", default=list(DEFAULT_TRAIN_VERSIONS)
    )
    parser.add_argument(
        "--holdout-versions", nargs="+", default=list(DEFAULT_HOLDOUT_VERSIONS)
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def extract_version(manifest: dict[str, Any]) -> str:
    for key in ("version", "source_version", "trajectory_version"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"manifest row has no version field: {sorted(manifest)}")


def extract_prompt(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("prompt", "prompt_messages", "messages"):
        value = manifest.get(key)
        if isinstance(value, list) and value:
            if not all(isinstance(item, dict) for item in value):
                break
            return value
    raise ValueError(f"manifest row has no prompt message list: {sorted(manifest)}")


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        normalized.append(
            {
                key: message[key]
                for key in ("role", "content", "tool_calls", "tool_call_id", "name")
                if key in message
            }
        )
    return normalized


def prompt_prefix_matches(messages: Any, prompt: list[dict[str, Any]]) -> bool:
    normalized_messages = normalize_messages(messages)
    normalized_prompt = normalize_messages(prompt)
    return normalized_messages[: len(normalized_prompt)] == normalized_prompt


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def counter_dict(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        value = first_value(row, keys)
        counter[str(value) if value is not None else "<missing>"] += 1
    return dict(sorted(counter.items()))


def nested_pair_counter(
    rows: list[dict[str, Any]], field: str, left: str = "chosen", right: str = "rejected"
) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        left_value = row.get(left, {}).get(field)
        right_value = row.get(right, {}).get(field)
        counter[f"{left_value or '<missing>'}>{right_value or '<missing>'}"] += 1
    return dict(sorted(counter.items()))


def length_direction_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        token_lengths = row.get("token_lengths", {})
        chosen = token_lengths.get("chosen")
        rejected = token_lengths.get("rejected")
        if not isinstance(chosen, int) or not isinstance(rejected, int):
            counter["<missing>"] += 1
        elif chosen > rejected:
            counter["chosen_longer"] += 1
        elif chosen < rejected:
            counter["rejected_longer"] += 1
        else:
            counter["tie"] += 1
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    train_versions = set(args.train_versions)
    holdout_versions = set(args.holdout_versions)
    if not train_versions or not holdout_versions:
        raise ValueError("train and heldout version sets must both be non-empty")
    if train_versions & holdout_versions:
        raise ValueError("train and heldout version sets overlap")

    source_rows = read_jsonl(args.source_dataset)
    manifest_rows = read_jsonl(args.source_manifest)
    if len(source_rows) != len(manifest_rows):
        raise ValueError(
            f"dataset/manifest row mismatch: {len(source_rows)} != {len(manifest_rows)}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")

    train_preferences: list[dict[str, Any]] = []
    train_sft: list[dict[str, Any]] = []
    train_manifest: list[dict[str, Any]] = []
    holdout_tasks: list[dict[str, Any]] = []
    holdout_manifest: list[dict[str, Any]] = []
    train_prompt_hashes: set[str] = set()
    holdout_prompt_hashes: set[str] = set()
    pair_ids: set[str] = set()

    for row_index, (preference, manifest) in enumerate(
        zip(source_rows, manifest_rows, strict=True)
    ):
        if "messages" not in preference or "rejected_messages" not in preference:
            raise ValueError(
                f"row {row_index}: expected messages and rejected_messages keys"
            )

        prompt = extract_prompt(manifest)
        if not prompt_prefix_matches(preference["messages"], prompt):
            raise ValueError(f"row {row_index}: chosen prompt prefix does not match")
        if not prompt_prefix_matches(preference["rejected_messages"], prompt):
            raise ValueError(f"row {row_index}: rejected prompt prefix does not match")

        prompt_hash = sha256_json(normalize_messages(prompt))
        pair_id = str(
            first_value(manifest, ("pair_id", "task_id", "id"))
            or f"source-row-{row_index:06d}"
        )
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair/task identifier: {pair_id}")
        pair_ids.add(pair_id)

        frozen_manifest = dict(manifest)
        frozen_manifest["_source_row"] = row_index
        frozen_manifest["_prompt_sha256"] = prompt_hash
        version = extract_version(manifest)

        if version in train_versions:
            train_preferences.append(preference)
            sft_row = {"messages": preference["messages"]}
            if "tools" in preference:
                sft_row["tools"] = preference["tools"]
            train_sft.append(sft_row)
            train_manifest.append(frozen_manifest)
            train_prompt_hashes.add(prompt_hash)
        elif version in holdout_versions:
            holdout_tasks.append(
                {
                    "prompt": prompt,
                    "tools": preference.get("tools"),
                    "metadata": {
                        "pair_id": pair_id,
                        "version": version,
                        "source_row": row_index,
                        "prompt_sha256": prompt_hash,
                    },
                }
            )
            holdout_manifest.append(frozen_manifest)
            holdout_prompt_hashes.add(prompt_hash)
        else:
            raise ValueError(f"row {row_index}: unexpected version {version!r}")

    overlap = sorted(train_prompt_hashes & holdout_prompt_hashes)
    if overlap:
        raise ValueError(f"train/heldout prompt leakage: {len(overlap)} overlapping hashes")
    if len(train_prompt_hashes) != len(train_preferences):
        raise ValueError("duplicate prompt detected in training split")
    if len(holdout_prompt_hashes) != len(holdout_tasks):
        raise ValueError("duplicate prompt detected in heldout split")

    randomized_indices = list(range(len(train_preferences)))
    random.Random(args.random_seed).shuffle(randomized_indices)
    swapped_indices = set(randomized_indices[: len(randomized_indices) // 2])
    randomized_preferences: list[dict[str, Any]] = []
    randomized_manifest: list[dict[str, Any]] = []
    for row_index, (preference, manifest) in enumerate(
        zip(train_preferences, train_manifest, strict=True)
    ):
        randomized = dict(preference)
        randomized_info = dict(manifest)
        is_swapped = row_index in swapped_indices
        if is_swapped:
            randomized["messages"] = preference["rejected_messages"]
            randomized["rejected_messages"] = preference["messages"]
        randomized_info["_randomized_label_swapped"] = is_swapped
        randomized_preferences.append(randomized)
        randomized_manifest.append(randomized_info)

    preference_worst_index = max(
        range(len(train_manifest)),
        key=lambda index: train_manifest[index]
        .get("token_lengths", {})
        .get("pair_max", -1),
    )
    sft_worst_index = max(
        range(len(train_manifest)),
        key=lambda index: train_manifest[index]
        .get("token_lengths", {})
        .get("chosen", -1),
    )

    outputs = {
        "train_preference.jsonl": train_preferences,
        "train_chosen_sft.jsonl": train_sft,
        "train_manifest.jsonl": train_manifest,
        f"train_preference_randomized_seed{args.random_seed}.jsonl": randomized_preferences,
        f"train_manifest_randomized_seed{args.random_seed}.jsonl": randomized_manifest,
        "smoke_preference_worst1.jsonl": [
            train_preferences[preference_worst_index]
        ],
        "smoke_preference_worst1_manifest.jsonl": [
            train_manifest[preference_worst_index]
        ],
        "smoke_chosen_sft_worst1.jsonl": [train_sft[sft_worst_index]],
        "smoke_chosen_sft_worst1_manifest.jsonl": [
            train_manifest[sft_worst_index]
        ],
        "holdout_tasks.jsonl": holdout_tasks,
        "holdout_manifest.jsonl": holdout_manifest,
    }
    for filename, rows in outputs.items():
        write_jsonl(output_dir / filename, rows)

    audit = {
        "experiment": "P-001",
        "split_policy": {
            "train_versions": sorted(train_versions),
            "holdout_versions": sorted(holdout_versions),
            "source_order_preserved": True,
            "randomized_label_seed": args.random_seed,
            "randomized_label_swapped": len(swapped_indices),
        },
        "source": {
            "dataset": str(args.source_dataset.resolve()),
            "dataset_sha256": sha256_file(args.source_dataset),
            "manifest": str(args.source_manifest.resolve()),
            "manifest_sha256": sha256_file(args.source_manifest),
            "rows": len(source_rows),
        },
        "counts": {
            "train": len(train_preferences),
            "holdout": len(holdout_tasks),
            "train_unique_prompts": len(train_prompt_hashes),
            "holdout_unique_prompts": len(holdout_prompt_hashes),
            "prompt_overlap": 0,
        },
        "train_distribution": {
            "version": counter_dict(
                train_manifest, ("version", "source_version", "trajectory_version")
            ),
            "task_type": counter_dict(
                train_manifest, ("task_type", "domain", "type", "dataset")
            ),
            "verdict_direction": nested_pair_counter(train_manifest, "verdict"),
            "model_direction": nested_pair_counter(train_manifest, "model"),
            "token_length_direction": length_direction_counter(train_manifest),
        },
        "holdout_distribution": {
            "version": counter_dict(
                holdout_manifest, ("version", "source_version", "trajectory_version")
            ),
            "task_type": counter_dict(
                holdout_manifest, ("task_type", "domain", "type", "dataset")
            ),
            "verdict_direction": nested_pair_counter(holdout_manifest, "verdict"),
            "model_direction": nested_pair_counter(holdout_manifest, "model"),
            "token_length_direction": length_direction_counter(holdout_manifest),
        },
        "outputs": {},
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit["outputs"] = {
        filename: {"rows": len(rows), "sha256": sha256_file(output_dir / filename)}
        for filename, rows in outputs.items()
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
