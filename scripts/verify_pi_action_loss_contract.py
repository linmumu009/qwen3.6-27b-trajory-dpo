#!/usr/bin/env python3
"""Verify with the real ms-swift template that observations have zero loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pi_trajectory_contract import annotate_action_loss


def encoded_contract(template: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
    encoded = template.encode({"messages": messages, "tools": tools})
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    if len(input_ids) != len(labels):
        raise RuntimeError("input_ids and labels have different lengths")
    tokenizer = template.tokenizer
    trainable_text = tokenizer.decode(
        [token for token, label in zip(input_ids, labels) if label != -100],
        skip_special_tokens=False,
    )
    masked_text = tokenizer.decode(
        [token for token, label in zip(input_ids, labels) if label == -100],
        skip_special_tokens=False,
    )
    checks = {
        "assistant_action_trainable": "ASSISTANT_ACTION_SENTINEL" in trainable_text,
        "tool_call_action_trainable": "TOOL_ACTION_SENTINEL" in trainable_text,
        "final_action_trainable": "FINAL_ACTION_SENTINEL" in trainable_text,
        "user_observation_masked": "USER_OBSERVATION_SENTINEL" not in trainable_text,
        "tool_result_observation_masked": (
            "TOOL_RESULT_OBSERVATION_SENTINEL" not in trainable_text
        ),
        "user_observation_present_in_context": (
            "USER_OBSERVATION_SENTINEL" in masked_text
        ),
        "tool_result_present_in_context": (
            "TOOL_RESULT_OBSERVATION_SENTINEL" in masked_text
        ),
    }
    return {
        "total_tokens": len(input_ids),
        "trainable_tokens": sum(label != -100 for label in labels),
        "masked_tokens": sum(label == -100 for label in labels),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.6-27B"))
    args = parser.parse_args()

    from swift import get_processor, get_template

    processor = get_processor(str(args.model))
    template = get_template(
        processor,
        max_length=8192,
        loss_scale="default",
        enable_thinking=True,
    )
    template.set_mode("train")

    messages = [
        {"role": "user", "content": "USER_OBSERVATION_SENTINEL"},
        {"role": "assistant", "content": "ASSISTANT_ACTION_SENTINEL"},
        {
            "role": "tool_call",
            "content": json.dumps(
                {
                    "name": "bash",
                    "arguments": {"command": "echo TOOL_ACTION_SENTINEL"},
                }
            ),
        },
        {"role": "tool_response", "content": "TOOL_RESULT_OBSERVATION_SENTINEL"},
        {"role": "assistant", "content": "FINAL_ACTION_SENTINEL"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    explicit_messages = annotate_action_loss(messages, generated_start=1)
    result = {
        "schema": "llin-pi-action-loss-contract-v1",
        "model": str(args.model),
        "default_template": encoded_contract(template, messages, tools),
        "explicit_message_contract": encoded_contract(
            template, explicit_messages, tools
        ),
    }
    result["passed"] = (
        result["default_template"]["passed"]
        and result["explicit_message_contract"]["passed"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
