from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_online_grpo_eval_prompts as prepare  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(prepare.stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_row(prompt_sha256: str = "eval-hash") -> dict:
    return {
        "prompt": [{"role": "user", "content": "question"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "metadata": {
            "prompt_sha256": prompt_sha256,
            "environment": {
                "environment_id": "sft/env-1",
                "task_id": "task-1",
            },
        },
    }


class PrepareOnlineGrpoEvalPromptsTest(unittest.TestCase):
    def test_freezes_executable_train_disjoint_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            verifiers = root / "verifiers.jsonl"
            train = root / "train.jsonl"
            output = root / "eval.jsonl"
            manifest = root / "freeze.json"
            write_jsonl(source, [source_row()])
            write_jsonl(
                verifiers,
                [
                    {
                        "verifier_id": "sft/env-1:task-1",
                        "environment_id": "sft/env-1",
                        "task_id": "task-1",
                    }
                ],
            )
            write_jsonl(train, [{"metadata": {"prompt_sha256": "train-hash"}}])

            argv = [
                "prepare_online_grpo_eval_prompts.py",
                str(source),
                str(verifiers),
                str(output),
                "--train-dataset",
                str(train),
                "--output-manifest",
                str(manifest),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(prepare.main(), 0)

            frozen = prepare.read_jsonl(output)
            self.assertEqual(len(frozen), 1)
            self.assertEqual(frozen[0]["messages"], source_row()["prompt"])
            self.assertEqual(frozen[0]["verifier_id"], "sft/env-1:task-1")
            self.assertEqual(
                frozen[0]["chat_template_kwargs"],
                {"enable_thinking": True},
            )
            self.assertEqual(
                frozen[0]["metadata"]["online_eval_source_index"],
                0,
            )
            evidence = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(evidence["rows"], 1)
            self.assertEqual(evidence["train_prompt_overlap"], 0)
            self.assertEqual(
                evidence["output_dataset_sha256"],
                prepare.file_sha256(output),
            )
            self.assertIn("not a fresh external heldout", evidence["claim_scope"])

    def test_rejects_train_eval_prompt_overlap_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            verifiers = root / "verifiers.jsonl"
            train = root / "train.jsonl"
            output = root / "eval.jsonl"
            write_jsonl(source, [source_row("same-hash")])
            write_jsonl(
                verifiers,
                [
                    {
                        "verifier_id": "sft/env-1:task-1",
                        "environment_id": "sft/env-1",
                        "task_id": "task-1",
                    }
                ],
            )
            write_jsonl(train, [{"metadata": {"prompt_sha256": "same-hash"}}])

            argv = [
                "prepare_online_grpo_eval_prompts.py",
                str(source),
                str(verifiers),
                str(output),
                "--train-dataset",
                str(train),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "train/eval prompt overlap"):
                    prepare.main()

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
