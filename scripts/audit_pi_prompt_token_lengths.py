#!/usr/bin/env python3
"""Measure PI rollout prompt token lengths without emitting prompt text."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from audit_online_grpo_reward_signal import load_jsonl, make_request, prompt_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.6-27B"))
    args = parser.parse_args()

    from swift import get_processor, get_template

    rows = load_jsonl(args.dataset.resolve())
    processor = get_processor(str(args.model.resolve()))
    template = get_template(
        processor,
        max_length=131072,
        loss_scale="default",
        enable_thinking=True,
    )
    lengths = []
    for index, row in enumerate(rows):
        request = make_request(row, f"prompt-length-{index}-{uuid.uuid4().hex}")
        task_id, _, _ = prompt_identity(row)
        encoded = template.encode(
            {
                "messages": request["messages"],
                "tools": request.get("tools"),
            }
        )
        lengths.append(
            {
                "prompt_index": index,
                "task_id": task_id,
                "tokens": len(encoded["input_ids"]),
            }
        )
    values = [row["tokens"] for row in lengths]
    summary = {
        "schema": "llin-pi-prompt-token-length-audit-v1",
        "prompt_count": len(lengths),
        "min_tokens": min(values),
        "max_tokens": max(values),
        "max_prompt_indices": [
            row["prompt_index"] for row in lengths if row["tokens"] == max(values)
        ],
        "per_prompt": lengths,
        "prompt_text_included": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
