from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pi_trajectory_contract import (  # noqa: E402
    annotate_action_loss,
    classify_tool_failure,
    observation_token_allowance,
    reward_decision,
)


def breakdown(**overrides):
    value = {
        "score": 0.3,
        "safe": True,
        "valid_tool_protocol": True,
        "successful_tool_use": True,
        "queried_required_tables": False,
        "has_final_answer": True,
        "gold_evidence": False,
    }
    value.update(overrides)
    return value


class PiTrajectoryContractTest(unittest.TestCase):
    def test_action_mask_excludes_prompt_and_tool_observation(self):
        messages = [
            {"role": "assistant", "content": "prompt example"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "analysis"},
            {"role": "tool_call", "content": "{}"},
            {"role": "tool_response", "content": "environment output"},
            {"role": "assistant", "content": "answer"},
        ]
        annotated = annotate_action_loss(messages, generated_start=2)
        self.assertEqual(
            [message["loss"] for message in annotated],
            ["", "", "1", "1", "", "1"],
        )
        self.assertNotIn("loss", messages[0])

    def test_truncated_text_is_not_a_terminal_success(self):
        decision = reward_decision(
            breakdown(queried_required_tables=True, gold_evidence=True),
            {"stopped_reason": "total_token_limit", "tool_events": []},
        )
        self.assertFalse(decision.terminal_answer)
        self.assertTrue(decision.truncated)
        self.assertFalse(decision.outcome_success)
        self.assertEqual(decision.outcome_only_reward, 0.0)
        self.assertEqual(decision.hybrid_reward, 0.0)

    def test_hybrid_reward_only_credits_verified_terminal_progress(self):
        progress = reward_decision(
            breakdown(queried_required_tables=True),
            {"stopped_reason": "final_answer", "tool_events": []},
        )
        success = reward_decision(
            breakdown(queried_required_tables=True, gold_evidence=True),
            {"stopped_reason": "final_answer", "tool_events": []},
        )
        shallow = reward_decision(
            breakdown(),
            {"stopped_reason": "final_answer", "tool_events": []},
        )
        self.assertTrue(progress.verified_progress)
        self.assertEqual(progress.hybrid_reward, 0.2)
        self.assertEqual(success.hybrid_reward, 1.0)
        self.assertEqual(shallow.hybrid_reward, 0.0)

    def test_tool_failure_responsibility_is_explicit(self):
        self.assertEqual(
            classify_tool_failure(
                {"ok": False, "response_preview": "PermissionError: network disabled"}
            ),
            "agent_policy_blocked",
        )
        self.assertEqual(
            classify_tool_failure(
                {"ok": False, "response_preview": "Command timed out after 60 seconds."}
            ),
            "ambiguous_timeout",
        )
        self.assertIsNone(classify_tool_failure({"ok": True, "response_preview": ""}))

    def test_observation_budget_preserves_policy_tokens(self):
        self.assertEqual(
            observation_token_allowance(
                total_limit=2048,
                policy_reserve=768,
                observation_limit=1024,
                per_tool_limit=384,
                policy_used=0,
                observation_used=0,
            ),
            384,
        )
        self.assertEqual(
            observation_token_allowance(
                total_limit=2048,
                policy_reserve=768,
                observation_limit=1024,
                per_tool_limit=384,
                policy_used=0,
                observation_used=1000,
            ),
            24,
        )
        self.assertEqual(
            observation_token_allowance(
                total_limit=2048,
                policy_reserve=768,
                observation_limit=1024,
                per_tool_limit=384,
                policy_used=700,
                observation_used=1024,
            ),
            0,
        )

    def test_observation_budget_preserves_forced_finalization_tokens(self):
        self.assertEqual(
            observation_token_allowance(
                total_limit=2048,
                policy_reserve=768,
                observation_limit=1024,
                per_tool_limit=384,
                policy_used=900,
                observation_used=600,
                finalization_reserve=512,
            ),
            0,
        )

    def test_failed_forced_finalization_is_truncated(self):
        decision = reward_decision(
            breakdown(queried_required_tables=True, gold_evidence=True),
            {"stopped_reason": "finalization_length", "tool_events": []},
        )
        self.assertTrue(decision.truncated)
        self.assertFalse(decision.terminal_answer)
        self.assertEqual(decision.hybrid_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
