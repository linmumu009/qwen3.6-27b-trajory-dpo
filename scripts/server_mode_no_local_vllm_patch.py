"""Allow MS-Swift's remote vLLM server mode without a local vLLM runtime.

The trainer only uses the HTTP client. The shared-file transport plugin replaces
the local HCCL communicator before training starts, so importing the full vLLM
runtime on the trainer host is unnecessary and can introduce a second CANN ABI.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from packaging.version import Version
from swift.rlhf_trainers import rollout_mixin
from swift.rlhf_trainers import vllm_client
from swift.megatron.trainers import rollout_mixin as megatron_rollout_mixin


if os.environ.get("LLIN_SHARED_LORA_SYNC_ROLE") != "train":
    raise RuntimeError("server_mode_no_local_vllm_patch.py is trainer-only")

REMOTE_VLLM_VERSION = os.environ["LLIN_REMOTE_VLLM_VERSION"]


def _available() -> bool:
    return True


def _remote_version_is_current(minimum: str) -> bool:
    # The rollout server is the source of truth for vLLM capabilities. The
    # trainer never imports or executes local vLLM kernels in this mode.
    return Version(REMOTE_VLLM_VERSION) >= Version(str(minimum))


vllm_client.is_vllm_available = _available
rollout_mixin.is_vllm_available = _available
rollout_mixin.check_vllm_version_ge = _remote_version_is_current
megatron_rollout_mixin.is_vllm_available = _available
megatron_rollout_mixin.check_vllm_version_ge = _remote_version_is_current

sync_file = Path(os.environ["LLIN_SHARED_LORA_SYNC_FILE"])
event_file = sync_file.with_suffix(
    f".server_mode_no_local_vllm.{os.getpid()}.jsonl"
)
event_file.parent.mkdir(parents=True, exist_ok=True)
with event_file.open("a", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            {
                "event": "remote_vllm_server_mode_enabled",
                "pid": os.getpid(),
                "time": time.time(),
                "local_vllm_runtime": False,
                "remote_vllm_version": REMOTE_VLLM_VERSION,
            },
            sort_keys=True,
        )
        + "\n"
    )
