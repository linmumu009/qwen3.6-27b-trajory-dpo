#!/usr/bin/env python3
"""Summarize completed reward-audit group files without emitting trajectory text."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def classify_tool_failure(preview: Any) -> str:
    text = preview.casefold() if isinstance(preview, str) else ""
    if "permissionerror" in text or "disabled" in text:
        return "policy_blocked"
    if "timed out" in text:
        return "timeout"
    if "jsondecodeerror" in text or "_parse_error" in text:
        return "malformed_arguments"
    if "command not found" in text:
        return "command_not_found"
    if "no such file" in text or "filenotfounderror" in text:
        return "missing_file"
    if "traceback" in text or "error" in text or "exception" in text:
        return "execution_error"
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups_dir = args.groups_dir.resolve()
    files = sorted(groups_dir.glob("prompt_*.json"))
    if not files:
        raise FileNotFoundError(f"no completed group files in {groups_dir}")
    groups = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if any(not isinstance(group, list) or len(group) != 8 for group in groups):
        raise RuntimeError("each completed group must contain exactly 8 samples")
    samples = [sample for group in groups for sample in group]

    reward_counts: collections.Counter[str] = collections.Counter()
    stopped_counts: collections.Counter[str] = collections.Counter()
    trajectory_contract_counts: collections.Counter[str] = collections.Counter()
    reward_contract_counts: collections.Counter[str] = collections.Counter()
    component_true: collections.Counter[str] = collections.Counter()
    component_false: collections.Counter[str] = collections.Counter()
    trajectory_v2_true: collections.Counter[str] = collections.Counter()
    trajectory_v2_false: collections.Counter[str] = collections.Counter()
    tool_failure_types: collections.Counter[str] = collections.Counter()
    rewards: list[float] = []
    elapsed: list[float] = []
    generated_tokens: list[float] = []
    policy_action_tokens: list[float] = []
    engine_reported_policy_tokens: list[float] = []
    fallback_policy_tokens: list[float] = []
    fallback_policy_token_turns: list[float] = []
    tool_response_tokens: list[float] = []
    observation_tokens: list[float] = []
    tool_calls: list[float] = []
    budget_hits = 0
    terminal_answers = 0
    finalization_triggered = 0
    finalization_succeeded = 0
    agent_failure_events = 0
    ambiguous_failure_events = 0
    per_prompt = []

    for group in groups:
        group_rewards = [float(sample["reward"]) for sample in group]
        per_prompt.append(
            {
                "prompt_index": int(group[0]["prompt_index"]),
                "task_id": group[0]["task_id"],
                "environment_id": group[0]["environment_id"],
                "rewards": group_rewards,
                "reward_mean": statistics.fmean(group_rewards),
                "reward_std_population": statistics.pstdev(group_rewards),
                "reward_min": min(group_rewards),
                "reward_max": max(group_rewards),
                "unique_reward_count": len(set(group_rewards)),
                "budget_hit_count": sum(bool(sample["budget_hit"]) for sample in group),
            }
        )

    for sample in samples:
        reward = float(sample["reward"])
        infos = sample["rollout_infos"]
        rewards.append(reward)
        reward_counts[str(reward)] += 1
        stopped_counts[str(infos.get("stopped_reason"))] += 1
        trajectory_contract_counts[str(infos.get("trajectory_contract"))] += 1
        reward_contract_counts[str(sample.get("reward_contract"))] += 1
        terminal_answers += int(infos.get("stopped_reason") == "final_answer")
        finalization_triggered += int(bool(infos.get("finalization_triggered")))
        finalization_succeeded += int(bool(infos.get("finalization_succeeded")))
        elapsed.append(float(infos.get("elapsed_seconds") or 0))
        generated_tokens.append(float(infos.get("generated_tokens") or 0))
        policy_action_tokens.append(float(infos.get("policy_action_tokens") or 0))
        engine_reported_policy_tokens.append(
            float(infos.get("engine_reported_policy_tokens") or 0)
        )
        fallback_policy_tokens.append(float(infos.get("fallback_policy_tokens") or 0))
        fallback_policy_token_turns.append(
            float(infos.get("fallback_policy_token_turns") or 0)
        )
        tool_response_tokens.append(float(infos.get("tool_response_tokens") or 0))
        observation_tokens.append(float(infos.get("observation_tokens") or 0))
        tool_calls.append(float(infos.get("tool_call_count") or 0))
        budget_hits += int(bool(sample["budget_hit"]))
        for key, value in sample["reward_breakdown"].items():
            if key == "score":
                continue
            (component_true if value else component_false)[key] += 1
        for key, value in (sample.get("trajectory_v2_reward") or {}).items():
            if isinstance(value, bool):
                (trajectory_v2_true if value else trajectory_v2_false)[key] += 1
        decision = sample.get("trajectory_v2_reward") or {}
        agent_failure_events += int(decision.get("agent_failure_count") or 0)
        ambiguous_failure_events += int(decision.get("ambiguous_failure_count") or 0)
        for event in infos.get("tool_events") or []:
            if not event.get("ok", False):
                tool_failure_types[classify_tool_failure(event.get("response_preview"))] += 1

    summary = {
        "schema": "llin-online-grpo-reward-signal-partial-audit-v1",
        "partial": True,
        "quality_claims_allowed": False,
        "policy_update_performed": False,
        "completed_prompt_groups": len(groups),
        "trajectories": len(samples),
        "reward": describe(rewards),
        "reward_counts": dict(sorted(reward_counts.items())),
        "reward_contract_counts": dict(sorted(reward_contract_counts.items())),
        "trajectory_contract_counts": dict(sorted(trajectory_contract_counts.items())),
        "nonzero_variance_groups": sum(
            row["reward_std_population"] > 0 for row in per_prompt
        ),
        "zero_variance_prompt_indices": [
            row["prompt_index"] for row in per_prompt if row["reward_std_population"] == 0
        ],
        "component_true_counts": dict(sorted(component_true.items())),
        "component_false_counts": dict(sorted(component_false.items())),
        "trajectory_v2_true_counts": dict(sorted(trajectory_v2_true.items())),
        "trajectory_v2_false_counts": dict(sorted(trajectory_v2_false.items())),
        "agent_failure_event_count": agent_failure_events,
        "ambiguous_failure_event_count": ambiguous_failure_events,
        "stopped_reason_counts": dict(sorted(stopped_counts.items())),
        "terminal_answer_count": terminal_answers,
        "finalization_triggered_count": finalization_triggered,
        "finalization_succeeded_count": finalization_succeeded,
        "tool_failure_type_counts": dict(sorted(tool_failure_types.items())),
        "budget_hit_count": budget_hits,
        "budget_hit_rate": budget_hits / len(samples),
        "elapsed_seconds": describe(elapsed),
        "generated_tokens": describe(generated_tokens),
        "policy_action_tokens": describe(policy_action_tokens),
        "engine_reported_policy_tokens": describe(engine_reported_policy_tokens),
        "fallback_policy_tokens": describe(fallback_policy_tokens),
        "fallback_policy_token_turns": describe(fallback_policy_token_turns),
        "tool_response_tokens": describe(tool_response_tokens),
        "observation_tokens": describe(observation_tokens),
        "tool_call_count": describe(tool_calls),
        "per_prompt": per_prompt,
        "group_file_sha256": {path.name: sha256(path) for path in files},
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
