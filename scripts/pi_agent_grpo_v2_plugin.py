#!/usr/bin/env python3
"""PI trajectory GRPO v2 scheduler and conservative hybrid reward plugin."""

from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from pi_trajectory_contract import (
    MASK_LOSS_MARKER,
    TRAIN_LOSS_MARKER,
    annotate_action_loss,
    observation_token_allowance,
    queue_masked_finalization,
    reward_decision,
)


V1_PLUGIN_PATH = Path(
    os.environ.get(
        "PI_AGENT_V1_PLUGIN",
        "/workspace/grpo_run/shared/pi_agent_grpo_plugin.py",
    )
).resolve()
if not V1_PLUGIN_PATH.is_file():
    raise FileNotFoundError(V1_PLUGIN_PATH)
if str(V1_PLUGIN_PATH.parent) not in sys.path:
    sys.path.insert(0, str(V1_PLUGIN_PATH.parent))
v1 = importlib.import_module(V1_PLUGIN_PATH.stem)


class PiAgentSchedulerV2(v1.PiAgentScheduler):
    """Separate policy-action and tool-observation budgets with explicit masks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_token_reserve = int(
            os.environ.get("PI_AGENT_POLICY_TOKEN_RESERVE", "768")
        )
        self.observation_token_limit = int(
            os.environ.get("PI_AGENT_OBSERVATION_TOKEN_LIMIT", "1024")
        )
        self.per_tool_observation_limit = int(
            os.environ.get("PI_AGENT_PER_TOOL_OBSERVATION_LIMIT", "384")
        )
        self.finalization_token_reserve = int(
            os.environ.get("PI_AGENT_FINALIZATION_TOKEN_RESERVE", "512")
        )
        self.per_turn_policy_token_limit = int(
            os.environ.get("PI_AGENT_PER_TURN_POLICY_TOKEN_LIMIT", "4096")
        )
        if self.finalization_token_reserve <= 0:
            raise ValueError("PI_AGENT_FINALIZATION_TOKEN_RESERVE must be positive")
        if self.per_turn_policy_token_limit <= 0:
            raise ValueError("PI_AGENT_PER_TURN_POLICY_TOKEN_LIMIT must be positive")
        observation_token_allowance(
            total_limit=self.total_token_limit,
            policy_reserve=self.policy_token_reserve,
            observation_limit=self.observation_token_limit,
            per_tool_limit=self.per_tool_observation_limit,
            policy_used=0,
            observation_used=0,
            finalization_reserve=self.finalization_token_reserve,
        )

    def _fallback_generated_token_count(
        self,
        raw_content: str,
        tool_calls: list[Any],
    ) -> int:
        fragments = [raw_content]
        if tool_calls and not v1.TOOL_BLOCK_PATTERN.search(raw_content):
            for call in tool_calls:
                fragments.append(
                    v1.stable_json(
                        {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                    )
                )
        return self._token_count("\n".join(fragments))

    async def run(
        self,
        infer_request,
        request_config: v1.RequestConfig,
        **kwargs,
    ) -> v1.RolloutOutput:
        generated_start = len(infer_request.messages)
        await self.on_trajectory_start([infer_request])
        current_turn = 1
        response = None
        stopped_reason = "unknown"
        policy_tokens = 0
        engine_reported_policy_tokens = 0
        fallback_policy_tokens = 0
        fallback_turns = 0
        observation_tokens = 0
        observation_exhausted = False
        finalization_pending = False
        finalization_triggered = False
        finalization_succeeded = False
        finalization_instruction = (
            "The environment observation budget is exhausted. "
            "Do not call any more tools. Give the best grounded final "
            "answer using only the evidence already observed."
        )

        while True:
            turn_config = copy.copy(request_config)
            remaining_total = (
                self.total_token_limit - policy_tokens - observation_tokens
            )
            if remaining_total <= 64 and response is not None:
                stopped_reason = "total_token_limit"
                break
            remaining_total = max(remaining_total, 1)
            if turn_config.max_tokens is None:
                turn_config.max_tokens = remaining_total
            else:
                turn_config.max_tokens = min(
                    turn_config.max_tokens, remaining_total
                )
            turn_config.max_tokens = min(
                turn_config.max_tokens,
                self.per_turn_policy_token_limit,
            )
            turn_request = infer_request
            if finalization_pending:
                turn_config.max_tokens = min(
                    turn_config.max_tokens,
                    self.finalization_token_reserve,
                )
                turn_request = copy.copy(infer_request)
                turn_request.tools = []

            response = await self.infer_engine.infer_async(
                turn_request, turn_config, **kwargs
            )
            choice = response.choices[0]
            raw_content = choice.message.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
            tool_calls = choice.message.tool_calls or []
            generated_ids = choice.token_ids or []
            if generated_ids:
                turn_policy_tokens = len(generated_ids)
                engine_reported_policy_tokens += turn_policy_tokens
            else:
                turn_policy_tokens = self._fallback_generated_token_count(
                    raw_content, tool_calls
                )
                fallback_policy_tokens += turn_policy_tokens
                fallback_turns += 1
            policy_tokens += turn_policy_tokens

            assistant_content = v1.TOOL_BLOCK_PATTERN.sub("\n", raw_content).strip()
            infer_request.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "loss": TRAIN_LOSS_MARKER,
                }
            )

            if finalization_pending:
                if choice.finish_reason == "length":
                    stopped_reason = "finalization_length"
                elif tool_calls:
                    stopped_reason = "finalization_tool_call"
                else:
                    stopped_reason = (
                        "final_answer" if assistant_content else "empty_final_answer"
                    )
                    finalization_succeeded = stopped_reason == "final_answer"
                break
            if choice.finish_reason == "length":
                stopped_reason = "length"
                break
            if not tool_calls:
                stopped_reason = (
                    "final_answer" if assistant_content else "empty_final_answer"
                )
                break
            if self.max_turns and current_turn >= self.max_turns:
                stopped_reason = "max_turns"
                break

            for call in tool_calls:
                available_observation_tokens = observation_token_allowance(
                    total_limit=self.total_token_limit,
                    policy_reserve=self.policy_token_reserve,
                    observation_limit=self.observation_token_limit,
                    per_tool_limit=self.per_tool_observation_limit,
                    policy_used=policy_tokens,
                    observation_used=observation_tokens,
                    finalization_reserve=self.finalization_token_reserve,
                )
                if available_observation_tokens <= 0:
                    observation_exhausted = True
                    break

                name = call.function.name
                try:
                    arguments = v1.parse_arguments(call.function.arguments)
                except Exception as exc:
                    arguments = {
                        "_parse_error": f"{type(exc).__name__}: {exc}"
                    }
                tool_payload = {"name": name, "arguments": arguments}
                infer_request.messages.append(
                    {
                        "role": "tool_call",
                        "content": v1.stable_json(tool_payload),
                        "loss": TRAIN_LOSS_MARKER,
                    }
                )
                if "_parse_error" in arguments:
                    result = v1.ToolResult(arguments["_parse_error"], False, 0.0)
                else:
                    result = await v1.asyncio.to_thread(
                        self._execute,
                        infer_request.uuid,
                        name,
                        arguments,
                    )
                result, result_tokens = self._cap_tool_result_to_budget(
                    result, available_observation_tokens
                )
                observation_tokens += result_tokens
                self._record_event(
                    infer_request.uuid,
                    turn=current_turn,
                    name=name,
                    arguments=arguments,
                    result=result,
                )
                infer_request.messages.append(
                    {
                        "role": "tool_response",
                        "content": result.content,
                        "loss": MASK_LOSS_MARKER,
                    }
                )

            if observation_exhausted:
                remaining_for_finalization = (
                    self.total_token_limit - policy_tokens - observation_tokens
                )
                if remaining_for_finalization <= 64:
                    stopped_reason = "observation_token_limit"
                    break
                queue_masked_finalization(
                    infer_request.messages,
                    finalization_instruction,
                )
                finalization_pending = True
                finalization_triggered = True
                current_turn += 1
                continue
            if observation_tokens >= self.observation_token_limit:
                remaining_for_finalization = (
                    self.total_token_limit - policy_tokens - observation_tokens
                )
                if remaining_for_finalization > 64:
                    queue_masked_finalization(
                        infer_request.messages,
                        finalization_instruction,
                    )
                    finalization_pending = True
                    finalization_triggered = True
                    current_turn += 1
                    continue
            if policy_tokens + observation_tokens >= self.total_token_limit - 64:
                stopped_reason = "total_token_limit"
                break
            current_turn += 1

        if response is None:
            raise RuntimeError("rollout produced no response")
        infer_request.messages = annotate_action_loss(
            infer_request.messages,
            generated_start=generated_start,
        )
        uuid = infer_request.uuid
        workspace = self._workspaces[uuid]
        environment = self._environment(infer_request)
        rollout_infos = {
            "trajectory_contract": "llin-pi-trajectory-grpo-v2",
            "num_turns": current_turn,
            "stopped_reason": stopped_reason,
            "has_terminal_answer": stopped_reason == "final_answer",
            "generated_tokens": policy_tokens,
            "policy_action_tokens": policy_tokens,
            "engine_reported_policy_tokens": engine_reported_policy_tokens,
            "fallback_policy_tokens": fallback_policy_tokens,
            "fallback_policy_token_turns": fallback_turns,
            "tool_response_tokens": observation_tokens,
            "observation_tokens": observation_tokens,
            "trajectory_budget_tokens": self.total_token_limit,
            "policy_token_reserve": self.policy_token_reserve,
            "observation_token_limit": self.observation_token_limit,
            "per_tool_observation_limit": self.per_tool_observation_limit,
            "finalization_token_reserve": self.finalization_token_reserve,
            "per_turn_policy_token_limit": self.per_turn_policy_token_limit,
            "finalization_triggered": finalization_triggered,
            "finalization_succeeded": finalization_succeeded,
            "tool_call_count": len(self._tool_events.get(uuid, [])),
            "tool_success_count": sum(
                event["ok"] for event in self._tool_events.get(uuid, [])
            ),
            "tool_events": self._tool_events.get(uuid, []),
            "sandbox_path": str(workspace),
            "environment_id": environment.get("environment_id"),
            "task_id": environment.get("task_id"),
            "elapsed_seconds": round(
                v1.time.monotonic() - self._started_at[uuid], 6
            ),
            "loss_contract": {
                "policy_action_roles": ["assistant", "tool_call"],
                "observation_roles": ["system", "user", "tool_response"],
                "observation_loss": 0,
            },
        }
        return v1.RolloutOutput(
            response=response,
            messages=infer_request.messages,
            response_token_ids=[],
            response_loss_mask=[],
            rollout_infos=rollout_infos,
            rollout_logprobs=[],
        )


class PiAgentTrajectoryV2ORM(v1.PiAgentTrajectoryORM):
    """Sparse terminal outcome plus conservative verified-progress reward."""

    def __call__(
        self,
        completions,
        messages=None,
        metadata=None,
        rollout_infos=None,
        **kwargs,
    ) -> list[float]:
        messages = messages or [[] for _ in completions]
        metadata = metadata or [{} for _ in completions]
        rollout_infos = rollout_infos or [{} for _ in completions]
        manifest = v1.load_manifest(self.manifest_path)
        rewards: list[float] = []
        for trajectory, meta, infos in zip(messages, metadata, rollout_infos):
            environment = (meta or {}).get("environment") or {}
            task_id = (infos or {}).get("task_id") or environment.get("task_id")
            environment_id = (infos or {}).get(
                "environment_id"
            ) or environment.get("environment_id")
            verifier_id = (meta or {}).get(
                "verifier_id"
            ) or f"{environment_id}:{task_id}"
            verifier = manifest.get(str(verifier_id)) or manifest.get(str(task_id))
            if verifier is None:
                rewards.append(0.0)
                continue
            workspace_value = (infos or {}).get("sandbox_path")
            workspace = (
                Path(workspace_value)
                if isinstance(workspace_value, str)
                else None
            )
            breakdown = v1.score_trajectory(trajectory, verifier, workspace)
            decision = reward_decision(
                v1.asdict(breakdown),
                infos or {},
                current_reward=breakdown.score,
                progress_reward=0.2,
            )
            if isinstance(infos, dict):
                infos["trajectory_v2_reward"] = decision.to_dict()
            rewards.append(decision.hybrid_reward)
        return rewards


v1.multi_turns["pi_agent_scheduler_v2"] = PiAgentSchedulerV2
v1.orms["pi_agent_trajectory_v2"] = PiAgentTrajectoryV2ORM
