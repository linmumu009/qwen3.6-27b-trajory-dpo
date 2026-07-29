#!/usr/bin/env python3
"""Sample grouped online PI trajectories and audit reward signal without training."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from pi_trajectory_contract import reward_decision


STANDARD_REQUEST_FIELDS = {
    "messages",
    "images",
    "audios",
    "videos",
    "tools",
    "objects",
    "chat_template_kwargs",
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_no}: row must be an object")
            rows.append(value)
    return rows


def load_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("llin_pi_agent_grpo_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def prompt_identity(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    metadata = row.get("metadata") or {}
    environment = metadata.get("environment") or {}
    return (
        environment.get("task_id"),
        environment.get("environment_id"),
        metadata.get("verifier_id") or row.get("verifier_id"),
    )


def make_request(row: dict[str, Any], request_uuid: str) -> dict[str, Any]:
    if not isinstance(row.get("messages"), list) or not row["messages"]:
        raise ValueError("dataset row is missing messages")
    request: dict[str, Any] = {
        "messages": copy.deepcopy(row["messages"]),
        "uuid": request_uuid,
    }
    for key in STANDARD_REQUEST_FIELDS - {"messages"}:
        if row.get(key) is not None:
            request[key] = copy.deepcopy(row[key])
    extra = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in STANDARD_REQUEST_FIELDS and key != "data_dict" and value is not None
    }
    base_data_dict = row.get("data_dict") or {}
    if not isinstance(base_data_dict, dict):
        raise TypeError("dataset data_dict must be an object")
    request["data_dict"] = {**extra, **copy.deepcopy(base_data_dict)}
    return request


def classify_tool_failure(preview: Any) -> str:
    text = preview.casefold() if isinstance(preview, str) else ""
    if "permissionerror" in text or "disabled" in text:
        return "policy_blocked"
    if "timed out" in text:
        return "timeout"
    if "_parse_error" in text or "jsondecodeerror" in text:
        return "malformed_arguments"
    if "command not found" in text:
        return "command_not_found"
    if "no such file" in text or "filenotfounderror" in text:
        return "missing_file"
    if "traceback" in text or "error" in text or "exception" in text:
        return "execution_error"
    return "other"


def sample_failure_reasons(sample: dict[str, Any]) -> list[str]:
    breakdown = sample["reward_breakdown"]
    infos = sample["rollout_infos"]
    reasons: list[str] = []
    mapping = (
        ("safe", "unsafe_or_no_bash"),
        ("valid_tool_protocol", "invalid_tool_protocol"),
        ("successful_tool_use", "no_successful_tool_response"),
        ("queried_required_tables", "required_tables_not_queried"),
        ("has_final_answer", "missing_final_answer"),
        ("gold_evidence", "gold_evidence_not_found"),
    )
    for field, reason in mapping:
        if not breakdown[field]:
            reasons.append(reason)
    stopped_reason = infos.get("stopped_reason")
    if stopped_reason in {
        "length",
        "total_token_limit",
        "observation_token_limit",
        "max_turns",
    }:
        reasons.append(f"stopped_{stopped_reason}")
    for event in infos.get("tool_events") or []:
        if not event.get("ok", False):
            reasons.append(f"tool_{classify_tool_failure(event.get('response_preview'))}")
    return sorted(set(reasons))


def group_diagnosis(samples: list[dict[str, Any]]) -> str:
    rewards = [float(sample["reward"]) for sample in samples]
    reward_values = len(set(rewards))
    trajectory_values = len({sample["trajectory_sha256"] for sample in samples})
    breakdown_values = len({stable_json(sample["reward_breakdown"]) for sample in samples})
    if reward_values > 1:
        return "nonzero_reward_variance"
    if rewards and rewards[0] == 1.0:
        return "uniformly_easy_under_current_verifier"
    if trajectory_values == 1:
        return "sampling_or_policy_homogeneity"
    if breakdown_values > 1:
        return "reward_collision_across_distinct_breakdowns"
    if rewards and rewards[0] <= 0.3:
        return "diverse_trajectories_uniformly_low_under_current_verifier"
    return "diverse_trajectories_uniform_verifier_breakdown"


def build_summary(
    *,
    samples: list[dict[str, Any]],
    dataset: Path,
    manifest: Path,
    plugin_path: Path,
    trajectories_path: Path,
    samples_per_prompt: int,
    request_config: dict[str, Any],
    base_url: str,
    started_at: str,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        grouped[int(sample["prompt_index"])].append(sample)

    failure_counts: collections.Counter[str] = collections.Counter()
    stopped_counts: collections.Counter[str] = collections.Counter()
    tool_failure_counts: collections.Counter[str] = collections.Counter()
    component_true_counts: collections.Counter[str] = collections.Counter()
    reward_counts: collections.Counter[str] = collections.Counter()
    generated_tokens: list[float] = []
    tool_response_tokens: list[float] = []
    total_budget_tokens: list[float] = []
    elapsed_seconds: list[float] = []
    budget_hits = 0
    per_prompt: list[dict[str, Any]] = []

    for prompt_index in sorted(grouped):
        group = grouped[prompt_index]
        if len(group) != samples_per_prompt:
            raise RuntimeError(
                f"prompt {prompt_index} has {len(group)} samples, expected {samples_per_prompt}"
            )
        rewards = [float(sample["reward"]) for sample in group]
        reward_std_population = statistics.pstdev(rewards)
        reward_std_sample = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        group_failures: collections.Counter[str] = collections.Counter()
        for sample in group:
            reasons = sample_failure_reasons(sample)
            group_failures.update(reasons)
            failure_counts.update(reasons)
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "task_id": group[0]["task_id"],
                "environment_id": group[0]["environment_id"],
                "samples": len(group),
                "rewards": rewards,
                "reward_mean": statistics.fmean(rewards),
                "reward_std_population": reward_std_population,
                "reward_std_sample": reward_std_sample,
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "unique_reward_count": len(set(rewards)),
                "unique_trajectory_count": len(
                    {sample["trajectory_sha256"] for sample in group}
                ),
                "unique_breakdown_count": len(
                    {stable_json(sample["reward_breakdown"]) for sample in group}
                ),
                "budget_hit_count": sum(sample["budget_hit"] for sample in group),
                "failure_reason_counts": dict(sorted(group_failures.items())),
                "diagnosis": group_diagnosis(group),
            }
        )

    for sample in samples:
        breakdown = sample["reward_breakdown"]
        infos = sample["rollout_infos"]
        reward_counts[str(sample["reward"])] += 1
        for key, value in breakdown.items():
            if key != "score" and value:
                component_true_counts[key] += 1
        stopped_counts[str(infos.get("stopped_reason"))] += 1
        generated = float(infos.get("generated_tokens") or 0)
        tool_tokens = float(infos.get("tool_response_tokens") or 0)
        generated_tokens.append(generated)
        tool_response_tokens.append(tool_tokens)
        total_budget_tokens.append(generated + tool_tokens)
        elapsed_seconds.append(float(infos.get("elapsed_seconds") or 0))
        budget_hits += int(sample["budget_hit"])
        for event in infos.get("tool_events") or []:
            if not event.get("ok", False):
                tool_failure_counts[classify_tool_failure(event.get("response_preview"))] += 1

    nonzero = [
        row["prompt_index"] for row in per_prompt if row["reward_std_population"] > 0
    ]
    uniform_easy = [
        row["prompt_index"]
        for row in per_prompt
        if row["diagnosis"] == "uniformly_easy_under_current_verifier"
    ]
    uniform_low = [
        row["prompt_index"]
        for row in per_prompt
        if row["diagnosis"] == "diverse_trajectories_uniformly_low_under_current_verifier"
    ]
    reward_values = [float(sample["reward"]) for sample in samples]
    return {
        "schema": "llin-online-grpo-reward-signal-audit-v1",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "quality_claims_allowed": False,
        "policy_update_performed": False,
        "inputs": {
            "dataset": str(dataset),
            "dataset_sha256": file_sha256(dataset),
            "verifier_manifest": str(manifest),
            "verifier_manifest_sha256": file_sha256(manifest),
            "plugin": str(plugin_path),
            "plugin_sha256": file_sha256(plugin_path),
            "base_url": base_url,
        },
        "sampling": {
            "prompt_count": len(grouped),
            "samples_per_prompt": samples_per_prompt,
            "trajectory_count": len(samples),
            "request_config": request_config,
        },
        "artifacts": {
            "trajectories_jsonl": str(trajectories_path),
            "trajectories_sha256": file_sha256(trajectories_path),
            "trajectories_bytes": trajectories_path.stat().st_size,
        },
        "aggregate": {
            "reward": describe(reward_values),
            "reward_counts": dict(sorted(reward_counts.items())),
            "component_true_counts": dict(sorted(component_true_counts.items())),
            "stopped_reason_counts": dict(sorted(stopped_counts.items())),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "tool_failure_type_counts": dict(sorted(tool_failure_counts.items())),
            "generated_tokens": describe(generated_tokens),
            "tool_response_tokens": describe(tool_response_tokens),
            "total_budget_tokens": describe(total_budget_tokens),
            "elapsed_seconds": describe(elapsed_seconds),
            "budget_hit_count": budget_hits,
            "budget_hit_rate": budget_hits / len(samples) if samples else None,
        },
        "per_prompt": per_prompt,
        "decision_gate": {
            "prompts_with_nonzero_reward_std": len(nonzero),
            "nonzero_reward_std_prompt_indices": nonzero,
            "uniformly_easy_prompt_indices": uniform_easy,
            "uniformly_low_prompt_indices": uniform_low,
            "engineering_variance_gate_pass": bool(nonzero),
            "training_authorized_by_this_audit": False,
            "reason": (
                "Nonzero group variance is necessary but verifier trust and candidate "
                "selection must be reviewed before any optimizer step."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("verifier_manifest", type=Path)
    parser.add_argument("plugin", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:28220")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--start-prompt", type=int, default=0)
    parser.add_argument("--end-prompt", type=int)
    parser.add_argument("--defer-summary", action="store_true")
    parser.add_argument(
        "--reward-contract",
        choices=("v1", "v2"),
        default="v1",
        help="v2 requires a terminal final_answer and uses conservative hybrid reward",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    manifest_path = args.verifier_manifest.resolve()
    plugin_path = args.plugin.resolve()
    output_dir = args.output_dir.resolve()
    if args.samples_per_prompt <= 1:
        raise ValueError("--samples-per-prompt must be greater than one")
    for path in (dataset, manifest_path, plugin_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = load_jsonl(dataset)
    if not rows:
        raise ValueError("dataset is empty")
    end_prompt = len(rows) if args.end_prompt is None else args.end_prompt
    if not 0 <= args.start_prompt < end_prompt <= len(rows):
        raise ValueError(
            f"invalid prompt range [{args.start_prompt}, {end_prompt}) for {len(rows)} rows"
        )
    plugin = load_plugin(plugin_path)
    manifest = plugin.load_manifest(str(manifest_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)
    started_at_path = output_dir / "started_at"
    if started_at_path.exists():
        started_at = started_at_path.read_text(encoding="utf-8").strip()
    else:
        started_at = datetime.now(timezone.utc).astimezone().isoformat()
        started_at_path.write_text(started_at + "\n", encoding="utf-8")

    request_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "n": 1,
        "logprobs": False,
    }
    session = requests.Session()
    endpoint = f"{args.base_url.rstrip('/')}/infer/"

    for prompt_index, row in enumerate(rows):
        if not args.start_prompt <= prompt_index < end_prompt:
            continue
        group_path = groups_dir / f"prompt_{prompt_index:03d}.json"
        if group_path.is_file():
            existing = json.loads(group_path.read_text(encoding="utf-8"))
            if len(existing) != args.samples_per_prompt:
                raise RuntimeError(f"incomplete existing group: {group_path}")
            print(
                stable_json(
                    {
                        "event": "prompt_skipped_complete",
                        "prompt_index": prompt_index,
                        "samples": len(existing),
                    }
                ),
                flush=True,
            )
            continue

        task_id, environment_id, verifier_id = prompt_identity(row)
        verifier = manifest.get(str(verifier_id)) or manifest.get(str(task_id))
        if verifier is None:
            raise KeyError(f"prompt {prompt_index}: no verifier for {verifier_id or task_id}")
        requests_for_group = [
            make_request(row, f"audit-{prompt_index:03d}-{sample_index:02d}-{uuid.uuid4().hex}")
            for sample_index in range(args.samples_per_prompt)
        ]
        started = time.monotonic()
        response = session.post(
            endpoint,
            json={
                "infer_requests": requests_for_group,
                "request_config": request_config,
                "use_tqdm": False,
            },
            timeout=args.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"prompt {prompt_index}: HTTP {response.status_code}: {response.text}"
            )
        outputs = response.json()
        if not isinstance(outputs, list) or len(outputs) != args.samples_per_prompt:
            raise RuntimeError(
                f"prompt {prompt_index}: expected {args.samples_per_prompt} outputs, "
                f"got {type(outputs).__name__}/{len(outputs) if isinstance(outputs, list) else '?'}"
            )

        group_samples: list[dict[str, Any]] = []
        for sample_index, (request, output) in enumerate(zip(requests_for_group, outputs)):
            messages = output.get("messages")
            infos = output.get("rollout_infos") or {}
            if not isinstance(messages, list):
                raise TypeError(f"prompt {prompt_index} sample {sample_index}: missing messages")
            workspace_value = infos.get("sandbox_path")
            workspace = Path(workspace_value) if isinstance(workspace_value, str) else None
            breakdown = asdict(plugin.score_trajectory(messages, verifier, workspace))
            decision = reward_decision(
                breakdown,
                infos,
                current_reward=float(breakdown["score"]),
                progress_reward=0.2,
            )
            reward = (
                float(breakdown["score"])
                if args.reward_contract == "v1"
                else decision.hybrid_reward
            )
            if reward < 0 or reward > 1:
                raise ValueError("reward outside [0, 1]")
            budget = int(infos.get("trajectory_budget_tokens") or args.max_tokens)
            used = int(infos.get("generated_tokens") or 0) + int(
                infos.get("tool_response_tokens") or 0
            )
            stopped_reason = infos.get("stopped_reason")
            budget_hit = stopped_reason in {"length", "total_token_limit"} or used >= budget - 64
            group_samples.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "task_id": task_id,
                    "environment_id": environment_id,
                    "verifier_id": verifier_id,
                    "request_uuid": request["uuid"],
                    "reward": reward,
                    "reward_contract": args.reward_contract,
                    "reward_breakdown": breakdown,
                    "trajectory_v2_reward": decision.to_dict(),
                    "rollout_infos": infos,
                    "budget_hit": budget_hit,
                    "trajectory_sha256": hashlib.sha256(
                        stable_json(messages).encode("utf-8")
                    ).hexdigest(),
                    "output": output,
                }
            )

        temporary = group_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(group_samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, group_path)
        print(
            stable_json(
                {
                    "event": "prompt_complete",
                    "prompt_index": prompt_index,
                    "samples": len(group_samples),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "rewards": [sample["reward"] for sample in group_samples],
                }
            ),
            flush=True,
        )

    if args.defer_summary:
        print(
            stable_json(
                {
                    "event": "audit_shard_complete",
                    "start_prompt": args.start_prompt,
                    "end_prompt": end_prompt,
                }
            ),
            flush=True,
        )
        return 0

    all_samples: list[dict[str, Any]] = []
    for prompt_index in range(len(rows)):
        group_path = groups_dir / f"prompt_{prompt_index:03d}.json"
        if not group_path.is_file():
            raise RuntimeError(f"missing completed group: {group_path}")
        group = json.loads(group_path.read_text(encoding="utf-8"))
        if len(group) != args.samples_per_prompt:
            raise RuntimeError(f"incomplete group: {group_path}")
        all_samples.extend(group)

    trajectories_path = output_dir / "trajectories.jsonl"
    with trajectories_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in all_samples:
            handle.write(stable_json(sample) + "\n")

    summary = build_summary(
        samples=all_samples,
        dataset=dataset,
        manifest=manifest_path,
        plugin_path=plugin_path,
        trajectories_path=trajectories_path,
        samples_per_prompt=args.samples_per_prompt,
        request_config=request_config,
        base_url=args.base_url,
        started_at=started_at,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        stable_json(
            {
                "event": "audit_complete",
                "summary": str(summary_path),
                "trajectories": len(all_samples),
                "prompts_with_nonzero_reward_std": summary["decision_gate"][
                    "prompts_with_nonzero_reward_std"
                ],
                "trajectories_sha256": summary["artifacts"]["trajectories_sha256"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
