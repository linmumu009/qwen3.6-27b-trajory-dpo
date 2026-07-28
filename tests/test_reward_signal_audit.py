from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_online_grpo_reward_signal as audit  # noqa: E402


def make_sample(prompt_index: int, sample_index: int, reward: float, trajectory: str):
    breakdown = {
        "score": reward,
        "safe": True,
        "valid_tool_protocol": True,
        "successful_tool_use": True,
        "queried_required_tables": reward >= 0.4,
        "has_final_answer": True,
        "gold_evidence": reward == 1.0,
    }
    return {
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "task_id": f"task-{prompt_index}",
        "environment_id": f"sft/env-{prompt_index}",
        "verifier_id": f"verifier-{prompt_index}",
        "request_uuid": f"request-{prompt_index}-{sample_index}",
        "reward": reward,
        "reward_breakdown": breakdown,
        "rollout_infos": {
            "stopped_reason": "final_answer",
            "generated_tokens": 100,
            "tool_response_tokens": 20,
            "trajectory_budget_tokens": 2048,
            "elapsed_seconds": 1.0,
            "tool_events": [],
        },
        "budget_hit": False,
        "trajectory_sha256": trajectory,
        "output": {"messages": []},
    }


class RewardSignalAuditTest(unittest.TestCase):
    def test_make_request_passes_dataset_metadata_via_data_dict(self):
        row = {
            "messages": [{"role": "user", "content": "question"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "metadata": {"verifier_id": "v1"},
            "verifier_id": "v1",
            "chat_template_kwargs": {"enable_thinking": True},
        }
        request = audit.make_request(row, "request-id")
        self.assertEqual(request["uuid"], "request-id")
        self.assertNotIn("metadata", {key: value for key, value in request.items() if key != "data_dict"})
        self.assertEqual(request["data_dict"]["metadata"], row["metadata"])
        self.assertEqual(request["data_dict"]["verifier_id"], "v1")
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": True})

    def test_group_diagnosis_separates_variance_and_uniform_low(self):
        varied = [
            make_sample(0, 0, 0.3, "a"),
            make_sample(0, 1, 1.0, "b"),
        ]
        uniform_low = [
            make_sample(0, 0, 0.3, "a"),
            make_sample(0, 1, 0.3, "b"),
        ]
        self.assertEqual(audit.group_diagnosis(varied), "nonzero_reward_variance")
        self.assertEqual(
            audit.group_diagnosis(uniform_low),
            "diverse_trajectories_uniformly_low_under_current_verifier",
        )

    def test_summary_opens_training_gate_only_for_nonzero_group_variance(self):
        samples = [
            make_sample(0, sample_index, 0.3 if sample_index < 4 else 1.0, f"a{sample_index}")
            for sample_index in range(8)
        ]
        samples += [
            make_sample(1, sample_index, 0.3, f"b{sample_index}")
            for sample_index in range(8)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            manifest = root / "manifest.jsonl"
            plugin = root / "plugin.py"
            trajectories = root / "trajectories.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            plugin.write_text("# fixture\n", encoding="utf-8")
            trajectories.write_text(
                "\n".join(audit.stable_json(sample) for sample in samples) + "\n",
                encoding="utf-8",
            )
            summary = audit.build_summary(
                samples=samples,
                dataset=dataset,
                manifest=manifest,
                plugin_path=plugin,
                trajectories_path=trajectories,
                samples_per_prompt=8,
                request_config={"temperature": 0.9},
                base_url="http://127.0.0.1:28220",
                started_at="2026-07-28T00:00:00+08:00",
            )
        self.assertTrue(summary["decision_gate"]["engineering_variance_gate_pass"])
        self.assertEqual(
            summary["decision_gate"]["nonzero_reward_std_prompt_indices"],
            [0],
        )
        self.assertFalse(summary["decision_gate"]["training_authorized_by_this_audit"])
        json.dumps(summary, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
