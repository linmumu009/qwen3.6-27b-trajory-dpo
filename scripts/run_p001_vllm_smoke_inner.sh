#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-/workspace/grpo_run/runs/p001_multihost_lora_smoke_20260728}"
HTTP_PORT="${2:-18080}"
ADAPTER_PATH="${3:-/workspace/grpo_run/shared/p001_multihost_handoff/adapters/chosen_sft_seed13_ckpt75_vllm_filtered}"
ADAPTER_NAME="${4:-p001-chosen-sft}"

test -r /models/Qwen3.6-27B/config.json
test -r "${ADAPTER_PATH}/adapter_config.json"
test -r "${ADAPTER_PATH}/adapter_model.safetensors"
mkdir -p "${RUN_DIR}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export PYTHONPATH="/workspace/grpo_run/shared/rollout_python_stubs${PYTHONPATH:+:${PYTHONPATH}}"
export LLIN_VLLM_ASCEND_IMPORT_ORDER_SHIM=1
export LLIN_VLLM_EXCLUDE_QWEN35_GDN_BA_LORA=1
export ATB_CXX_ABI=1
export SOC_VERSION=ascend910_9391
export TASK_QUEUE_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

{
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'http_port=%s\n' "${HTTP_PORT}"
  printf 'base_model=%s\n' /models/Qwen3.6-27B
  printf 'adapter_name=%s\n' "${ADAPTER_NAME}"
  printf 'adapter_path=%s\n' "${ADAPTER_PATH}"
  printf 'topology=%s\n' tp8
  python -V
  python -c 'import torch, vllm; print(f"torch={torch.__version__}"); print(f"vllm={vllm.__version__}"); print(f"npu_count={torch.npu.device_count()}")'
  sha256sum \
    "${ADAPTER_PATH}/adapter_config.json" \
    "${ADAPTER_PATH}/adapter_model.safetensors"
} >"${RUN_DIR}/environment_summary.txt"

exec python -c \
  'import vllm_ascend.ops.fused_moe.fused_moe; import runpy; runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")' \
  --model /models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --host 127.0.0.1 \
  --port "${HTTP_PORT}" \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}"
