"""Trainer-side cross-host transport for the existing shared-file LoRA patch."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from swift.rlhf_trainers import vllm_client as client_mod


def _sync_path() -> Path:
    value = os.environ.get("LLIN_SHARED_LORA_SYNC_FILE")
    if not value:
        raise RuntimeError("LLIN_SHARED_LORA_SYNC_FILE is required")
    path = Path(value)
    if not str(path).startswith("/workspace/grpo_run/runs/"):
        raise RuntimeError(f"Refusing non-run LoRA path: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _append_event(path: Path, event: dict) -> None:
    event_path = path.with_suffix(f".cross_host_trainer.{os.getpid()}.jsonl")
    event = {"time": time.time(), "pid": os.getpid(), **event}
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def update_adapter_flattened_cross_host(
    self, peft_config, metadatas, flattened_tensor
):
    path = _sync_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    peft_dict = client_mod.peft_config_to_dict(peft_config)
    metadata_dicts = [
        item.model_dump()
        if hasattr(item, "model_dump")
        else item.dict()
        if hasattr(item, "dict")
        else dict(item)
        for item in metadatas
    ]
    payload = {
        "format": "llin-ms-swift-flat-lora-v1",
        "created_at": time.time(),
        "peft_config": peft_dict,
        "metadatas": metadata_dicts,
        "flattened_tensor": flattened_tensor.detach()
        .contiguous()
        .to(device="cpu"),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)

    transfer_id = f"{time.time_ns()}-{os.getpid()}"
    digest = _sha256(path)
    size = path.stat().st_size
    request_path = path.with_suffix(".request.json")
    ack_path = path.with_suffix(".ack.json")
    request = {
        "format": "llin-cross-host-lora-sync-request-v1",
        "transfer_id": transfer_id,
        "sha256": digest,
        "bytes": size,
        "tensor_count": len(metadata_dicts),
        "created_at": time.time(),
    }
    _write_json_atomic(request_path, request)
    _append_event(path, {"event": "transfer_requested", **request})

    timeout = float(os.environ.get("LLIN_CROSS_HOST_SYNC_TIMEOUT", "300"))
    deadline = time.monotonic() + timeout
    ack = None
    while time.monotonic() < deadline:
        try:
            candidate = json.loads(ack_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        if (
            candidate.get("transfer_id") == transfer_id
            and candidate.get("sha256") == digest
            and candidate.get("bytes") == size
        ):
            ack = candidate
            break
        time.sleep(0.1)
    if ack is None:
        raise TimeoutError(
            f"Cross-host LoRA transfer was not acknowledged in {timeout}s: "
            f"{transfer_id}"
        )
    _append_event(path, {"event": "transfer_acknowledged", **ack})

    data = {"peft_config": peft_dict, "metadatas": metadata_dicts}
    errors = []
    for index in range(self.num_servers):
        try:
            response = self.sessions[index].post(
                f"{self.base_urls[index]}/update_adapter_flattened_param/",
                json=data,
                timeout=getattr(self, "connection_timeout", 1800),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"server {index}: {response.status_code} {response.text}"
                )
            body = response.json()
            if not body.get("all_workers_loaded", False):
                raise RuntimeError(
                    f"server {index} did not confirm all workers: {body}"
                )
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"Cross-host LoRA update failed: {errors}")
    _append_event(
        path,
        {
            "event": "all_rollout_workers_loaded",
            "transfer_id": transfer_id,
            "sha256": digest,
            "server_count": self.num_servers,
        },
    )


if os.environ.get("LLIN_SHARED_LORA_SYNC_ROLE") != "train":
    raise RuntimeError("cross_host_lora_sync_patch.py is trainer-only")

client_mod.VLLMClient.update_adapter_flattened_param = (
    update_adapter_flattened_cross_host
)
