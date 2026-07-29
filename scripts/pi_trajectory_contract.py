#!/usr/bin/env python3
"""Shared contracts for PI trajectory action masking and reward responsibility."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


ACTION_ROLES = frozenset({"assistant", "tool_call"})
OBSERVATION_ROLES = frozenset({"system", "user", "tool_response"})
TRAIN_LOSS_MARKER = "1"
MASK_LOSS_MARKER = ""
TERMINAL_STOP_REASON = "final_answer"
TRUNCATED_STOP_REASONS = frozenset(
    {
        "length",
        "max_turns",
        "observation_token_limit",
        "total_token_limit",
        "finalization_length",
        "finalization_tool_call",
    }
)


def role_should_train(role: str) -> bool:
    """Return whether tokens from a generated message role are policy actions."""
    if role in ACTION_ROLES:
        return True
    if role in OBSERVATION_ROLES:
        return False
    raise ValueError(f"unsupported trajectory role: {role!r}")


def annotate_action_loss(
    messages: Sequence[Mapping[str, Any]],
    *,
    generated_start: int,
) -> list[dict[str, Any]]:
    """Copy messages and explicitly mark only generated policy actions for loss.

    Messages before ``generated_start`` are prompt/history and are always masked,
    including any assistant examples that may be present in a future dataset.
    """
    if not 0 <= generated_start <= len(messages):
        raise ValueError(
            f"generated_start={generated_start} outside [0, {len(messages)}]"
        )
    annotated = copy.deepcopy(list(messages))
    for index, message in enumerate(annotated):
        role = message.get("role")
        if not isinstance(role, str):
            raise TypeError(f"message {index} is missing a string role")
        should_train = index >= generated_start and role_should_train(role)
        # This ms-swift revision applies truthiness to ``loss`` but its HTTP
        # response schema accepts strings/lists, not booleans.
        message["loss"] = TRAIN_LOSS_MARKER if should_train else MASK_LOSS_MARKER
    return annotated


def queue_masked_finalization(
    messages: list[dict[str, Any]],
    instruction: str,
) -> None:
    """Queue a masked final-answer instruction without breaking tool dialogue.

    A tool response must be followed by the assistant's next generation.  Adding
    a separate user message after a tool response makes the ms-swift template
    treat that user message as a response role and reject the request.  Merge
    the instruction into the masked observation in that case; after a policy
    action, a regular masked user message remains valid.
    """
    if not instruction.strip():
        raise ValueError("finalization instruction must not be empty")
    if messages and messages[-1].get("role") in {"tool_response", "user"}:
        previous = messages[-1].get("content")
        previous_text = previous if isinstance(previous, str) else str(previous or "")
        messages[-1]["content"] = f"{previous_text}\n\n{instruction}".strip()
        messages[-1]["loss"] = MASK_LOSS_MARKER
        return
    messages.append(
        {
            "role": "user",
            "content": instruction,
            "loss": MASK_LOSS_MARKER,
        }
    )


def observation_token_allowance(
    *,
    total_limit: int,
    policy_reserve: int,
    observation_limit: int,
    per_tool_limit: int,
    policy_used: int,
    observation_used: int,
    finalization_reserve: int = 0,
    safety_margin: int = 64,
) -> int:
    """Return the maximum tokens available to the next tool observation.

    The allowance enforces all three constraints simultaneously:

    - the total trajectory interaction budget;
    - a cumulative observation budget;
    - a protected minimum budget for future policy actions.
    """
    values = {
        "total_limit": total_limit,
        "policy_reserve": policy_reserve,
        "observation_limit": observation_limit,
        "per_tool_limit": per_tool_limit,
        "policy_used": policy_used,
        "observation_used": observation_used,
        "finalization_reserve": finalization_reserve,
        "safety_margin": safety_margin,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"token budget values must be non-negative: {values}")
    if observation_limit + finalization_reserve > total_limit - safety_margin:
        raise ValueError(
            "observation_limit + finalization_reserve must fit inside total_limit "
            "after safety_margin"
        )
    remaining_total = max(
        total_limit - policy_used - observation_used - safety_margin, 0
    )
    protected_policy = max(
        policy_reserve - policy_used,
        finalization_reserve,
    )
    available_after_reserve = max(remaining_total - protected_policy, 0)
    remaining_observation = max(observation_limit - observation_used, 0)
    return min(per_tool_limit, remaining_observation, available_after_reserve)


def classify_tool_failure(event: Mapping[str, Any]) -> str | None:
    """Classify a failed tool event without exposing its raw response text."""
    if event.get("ok", False):
        return None
    preview = event.get("response_preview")
    text = preview.casefold() if isinstance(preview, str) else ""
    if "permissionerror" in text or "disabled" in text:
        return "agent_policy_blocked"
    if "jsondecodeerror" in text or "_parse_error" in text:
        return "agent_malformed_arguments"
    if "command not found" in text:
        return "agent_command_not_found"
    if "no such file" in text or "filenotfounderror" in text:
        return "agent_missing_resource"
    if "timed out" in text:
        return "ambiguous_timeout"
    if "traceback" in text or "error" in text or "exception" in text:
        return "agent_execution_error"
    return "ambiguous_other"


@dataclass(frozen=True)
class RewardDecision:
    """Sparse reward variants plus dense, auditable evaluation criteria."""

    current_reward: float
    outcome_only_reward: float
    hybrid_reward: float
    outcome_success: bool
    verified_progress: bool
    terminal_answer: bool
    truncated: bool
    safe: bool
    valid_tool_protocol: bool
    successful_tool_use: bool
    queried_required_tables: bool
    gold_evidence: bool
    agent_failure_count: int
    ambiguous_failure_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reward_decision(
    reward_breakdown: Mapping[str, Any],
    rollout_infos: Mapping[str, Any],
    *,
    current_reward: float | None = None,
    progress_reward: float = 0.2,
) -> RewardDecision:
    """Compute outcome-only and conservative hybrid reward counterfactuals.

    The hybrid reward is intentionally sparse:

    - 1.0 for a terminal, safe, grounded success;
    - ``progress_reward`` for a terminal answer after successful required-table
      acquisition but without verified gold evidence;
    - 0.0 otherwise.

    Tool success, answer presence, or safe syntax alone do not earn reward.
    """
    if not 0.0 <= progress_reward < 1.0:
        raise ValueError("progress_reward must be in [0, 1)")
    required_fields = {
        "safe",
        "valid_tool_protocol",
        "successful_tool_use",
        "queried_required_tables",
        "gold_evidence",
    }
    missing = sorted(required_fields - reward_breakdown.keys())
    if missing:
        raise KeyError(f"reward_breakdown missing fields: {missing}")

    stopped_reason = str(rollout_infos.get("stopped_reason"))
    terminal_answer = stopped_reason == TERMINAL_STOP_REASON
    truncated = stopped_reason in TRUNCATED_STOP_REASONS
    safe = bool(reward_breakdown["safe"])
    valid_protocol = bool(reward_breakdown["valid_tool_protocol"])
    successful = bool(reward_breakdown["successful_tool_use"])
    queried = bool(reward_breakdown["queried_required_tables"])
    gold = bool(reward_breakdown["gold_evidence"])

    grounded_prefix = safe and valid_protocol and successful and queried and terminal_answer
    outcome_success = grounded_prefix and gold
    verified_progress = grounded_prefix and not gold

    failures = [
        classify_tool_failure(event)
        for event in (rollout_infos.get("tool_events") or [])
    ]
    agent_failures = sum(
        isinstance(value, str) and value.startswith("agent_") for value in failures
    )
    ambiguous_failures = sum(
        isinstance(value, str) and value.startswith("ambiguous_") for value in failures
    )
    if current_reward is None:
        current_reward = float(reward_breakdown.get("score", 0.0))

    return RewardDecision(
        current_reward=float(current_reward),
        outcome_only_reward=1.0 if outcome_success else 0.0,
        hybrid_reward=(
            1.0 if outcome_success else progress_reward if verified_progress else 0.0
        ),
        outcome_success=outcome_success,
        verified_progress=verified_progress,
        terminal_answer=terminal_answer,
        truncated=truncated,
        safe=safe,
        valid_tool_protocol=valid_protocol,
        successful_tool_use=successful,
        queried_required_tables=queried,
        gold_evidence=gold,
        agent_failure_count=agent_failures,
        ambiguous_failure_count=ambiguous_failures,
    )
