#!/usr/bin/env python3
"""Replay completed PI reward groups under trajectory-v2 reward contracts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from pi_trajectory_contract import reward_decision


VARIANTS = ("current_reward", "outcome_only_reward", "hybrid_reward")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-reward", type=float, default=0.2)
    args = parser.parse_args()

    groups_dir = args.groups_dir.resolve()
    files = sorted(groups_dir.glob("prompt_*.json"))
    if not files:
        raise FileNotFoundError(f"no completed group files in {groups_dir}")
    groups = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if any(not isinstance(group, list) or len(group) != 8 for group in groups):
        raise RuntimeError("each completed group must contain exactly 8 samples")

    variant_values: dict[str, list[float]] = {key: [] for key in VARIANTS}
    variant_counts: dict[str, collections.Counter[str]] = {
        key: collections.Counter() for key in VARIANTS
    }
    component_true: collections.Counter[str] = collections.Counter()
    component_false: collections.Counter[str] = collections.Counter()
    failure_classes: collections.Counter[str] = collections.Counter()
    stopped_reasons: collections.Counter[str] = collections.Counter()
    per_prompt: list[dict[str, Any]] = []
    outcome_successes = 0
    verified_progresses = 0
    terminal_answers = 0
    truncated = 0

    for group in groups:
        decisions = [
            reward_decision(
                sample["reward_breakdown"],
                sample["rollout_infos"],
                current_reward=float(sample["reward"]),
                progress_reward=args.progress_reward,
            )
            for sample in group
        ]
        row: dict[str, Any] = {
            "prompt_index": int(group[0]["prompt_index"]),
            "task_id": group[0]["task_id"],
            "environment_id": group[0]["environment_id"],
            "terminal_answer_count": sum(item.terminal_answer for item in decisions),
            "truncated_count": sum(item.truncated for item in decisions),
            "outcome_success_count": sum(item.outcome_success for item in decisions),
            "verified_progress_count": sum(item.verified_progress for item in decisions),
            "variants": {},
        }
        for variant in VARIANTS:
            rewards = [float(getattr(item, variant)) for item in decisions]
            row["variants"][variant] = {
                "rewards": rewards,
                "mean": statistics.fmean(rewards),
                "std_population": statistics.pstdev(rewards),
                "min": min(rewards),
                "max": max(rewards),
                "unique_reward_count": len(set(rewards)),
            }
            variant_values[variant].extend(rewards)
            variant_counts[variant].update(str(value) for value in rewards)
        per_prompt.append(row)

        for sample, decision in zip(group, decisions):
            outcome_successes += int(decision.outcome_success)
            verified_progresses += int(decision.verified_progress)
            terminal_answers += int(decision.terminal_answer)
            truncated += int(decision.truncated)
            stopped_reasons[str(sample["rollout_infos"].get("stopped_reason"))] += 1
            for key in (
                "safe",
                "valid_tool_protocol",
                "successful_tool_use",
                "queried_required_tables",
                "gold_evidence",
            ):
                (component_true if getattr(decision, key) else component_false)[key] += 1
            for event in sample["rollout_infos"].get("tool_events") or []:
                from pi_trajectory_contract import classify_tool_failure

                failure_class = classify_tool_failure(event)
                if failure_class is not None:
                    failure_classes[failure_class] += 1

    variant_summary: dict[str, Any] = {}
    for variant in VARIANTS:
        nonzero = [
            row["prompt_index"]
            for row in per_prompt
            if row["variants"][variant]["std_population"] > 0
        ]
        zero = [
            row["prompt_index"]
            for row in per_prompt
            if row["variants"][variant]["std_population"] == 0
        ]
        variant_summary[variant] = {
            "reward": describe(variant_values[variant]),
            "reward_counts": dict(sorted(variant_counts[variant].items())),
            "nonzero_variance_group_count": len(nonzero),
            "nonzero_variance_prompt_indices": nonzero,
            "zero_variance_prompt_indices": zero,
        }

    summary = {
        "schema": "llin-pi-trajectory-grpo-v2-counterfactual-audit-v1",
        "partial": True,
        "quality_claims_allowed": False,
        "policy_update_performed": False,
        "completed_prompt_groups": len(groups),
        "trajectories": sum(len(group) for group in groups),
        "contract": {
            "terminal_answer_requires_stopped_reason": "final_answer",
            "observation_tokens_participate_in_policy_loss": False,
            "tool_results_may_be_used_as_reward_evidence": True,
            "progress_reward": args.progress_reward,
            "hybrid_reward_definition": (
                "1.0 terminal safe grounded gold success; "
                "progress_reward terminal safe successful required-table progress; "
                "0.0 otherwise"
            ),
        },
        "terminal_answer_count": terminal_answers,
        "truncated_count": truncated,
        "outcome_success_count": outcome_successes,
        "verified_progress_count": verified_progresses,
        "component_true_counts": dict(sorted(component_true.items())),
        "component_false_counts": dict(sorted(component_false.items())),
        "stopped_reason_counts": dict(sorted(stopped_reasons.items())),
        "tool_failure_responsibility_counts": dict(sorted(failure_classes.items())),
        "variants": variant_summary,
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
