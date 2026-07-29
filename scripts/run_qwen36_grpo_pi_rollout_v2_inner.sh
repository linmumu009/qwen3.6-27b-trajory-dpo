#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s <run_name> <max_model_len> <completion_budget> <http_port>\n' "$0" >&2
  exit 2
fi

RUN_NAME="$1"
MAX_MODEL_LEN="$2"
COMPLETION_BUDGET="$3"
HTTP_PORT="$4"
RUN_DIR="/workspace/grpo_run/runs/${RUN_NAME}"
V1_PLUGIN=/workspace/grpo_run/shared/pi_agent_grpo_plugin.py
V2_PLUGIN=/workspace/grpo_run/shared/pi_agent_grpo_v2_plugin.py
CONTRACT=/workspace/grpo_run/shared/pi_trajectory_contract.py
SYNC_PLUGIN=/workspace/grpo_run/shared/shared_file_lora_sync_patch.py
WORKER_HOOK=/workspace/grpo_run/shared/rollout_worker_hook.py

test -r /models/Qwen3.6-27B/config.json
test -r "${V1_PLUGIN}"
test -r "${V2_PLUGIN}"
test -r "${CONTRACT}"
test -r "${SYNC_PLUGIN}"
test -r "${WORKER_HOOK}"
mkdir -p "${RUN_DIR}"

ROLLOUT_MODULE=/workspace/ms-swift/swift/pipelines/infer/rollout.py
if mountpoint -q "${ROLLOUT_MODULE}"; then
  umount "${ROLLOUT_MODULE}"
fi
mount --bind "${WORKER_HOOK}" "${ROLLOUT_MODULE}"
mount -o remount,bind,ro "${ROLLOUT_MODULE}"

export PYTHONPATH=/workspace/grpo_run/shared:/workspace/grpo_run/shared/rollout_python_stubs:/workspace/ms-swift
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export LLIN_VLLM_ASCEND_IMPORT_ORDER_SHIM=1
export LLIN_VLLM_EXCLUDE_QWEN35_GDN_BA_LORA=1
export VLLM_NO_USAGE_STATS=1
export PI_AGENT_SANDBOX_LOWER=/pi_sandbox_lower
export PI_AGENT_SANDBOX_RUN_ROOT=/workspace/grpo_run/rollout_sandboxes
export PI_AGENT_RUN_TAG="${RUN_NAME}"
export PI_AGENT_TOTAL_TOKEN_LIMIT="${COMPLETION_BUDGET}"
export PI_AGENT_POLICY_TOKEN_RESERVE="${PI_AGENT_POLICY_TOKEN_RESERVE:-768}"
export PI_AGENT_OBSERVATION_TOKEN_LIMIT="${PI_AGENT_OBSERVATION_TOKEN_LIMIT:-1024}"
export PI_AGENT_PER_TOOL_OBSERVATION_LIMIT="${PI_AGENT_PER_TOOL_OBSERVATION_LIMIT:-384}"
export PI_AGENT_MAX_TOOL_TIMEOUT=60
export PI_AGENT_MAX_TOOL_OUTPUT=200000
export PI_AGENT_COPY_CONCURRENCY=2
export PI_AGENT_V1_PLUGIN="${V1_PLUGIN}"
export LLIN_SHARED_LORA_SYNC_FILE="/workspace/grpo_run/runs/${RUN_NAME}/shared_lora_sync/adapter_flattened.pt"
export LLIN_SHARED_LORA_SYNC_ROLE=rollout

if (( PI_AGENT_POLICY_TOKEN_RESERVE + PI_AGENT_OBSERVATION_TOKEN_LIMIT > COMPLETION_BUDGET - 64 )); then
  printf 'invalid_v2_budget reserve=%s observation=%s total=%s\n' \
    "${PI_AGENT_POLICY_TOKEN_RESERVE}" \
    "${PI_AGENT_OBSERVATION_TOKEN_LIMIT}" \
    "${COMPLETION_BUDGET}" >&2
  exit 3
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

{
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'max_model_len=%s\n' "${MAX_MODEL_LEN}"
  printf 'completion_budget=%s\n' "${COMPLETION_BUDGET}"
  printf 'policy_token_reserve=%s\n' "${PI_AGENT_POLICY_TOKEN_RESERVE}"
  printf 'observation_token_limit=%s\n' "${PI_AGENT_OBSERVATION_TOKEN_LIMIT}"
  printf 'per_tool_observation_limit=%s\n' "${PI_AGENT_PER_TOOL_OBSERVATION_LIMIT}"
  printf 'http_port=%s\n' "${HTTP_PORT}"
  printf 'rollout_topology=tp4_dp2\n'
  printf 'trajectory_contract=llin-pi-trajectory-grpo-v2\n'
  printf 'loss_contract=assistant_and_tool_call_only\n'
  printf 'reward_contract=sparse_terminal_outcome_plus_verified_progress\n'
  printf 'worker_hook_sha256='
  sha256sum "${ROLLOUT_MODULE}" | awk '{print $1}'
  printf 'speculative_mtp=disabled_for_initial_dynamic_lora_sync\n'
  printf 'prefix_caching=true\n'
  printf 'dynamic_lora_transport=atomic_shared_file_cross_cann\n'
  python -V
  sha256sum \
    /models/Qwen3.6-27B/config.json \
    "${V1_PLUGIN}" \
    "${V2_PLUGIN}" \
    "${CONTRACT}" \
    "${SYNC_PLUGIN}"
} >"${RUN_DIR}/rollout_environment_summary.txt"

python -c 'import vllm_ascend.ops.fused_moe.fused_moe; import runpy; runpy.run_module("swift.cli.rollout", run_name="__main__")' \
  --model /models/Qwen3.6-27B \
  --infer_backend vllm \
  --host 127.0.0.1 \
  --port "${HTTP_PORT}" \
  --served_model_name Qwen3.6-27B \
  --vllm_tensor_parallel_size 4 \
  --vllm_data_parallel_size 2 \
  --vllm_max_model_len "${MAX_MODEL_LEN}" \
  --vllm_max_num_seqs 16 \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_enforce_eager true \
  --vllm_enable_prefix_caching true \
  --vllm_enable_lora true \
  --vllm_max_lora_rank 8 \
  --vllm_engine_kwargs '{"max_num_batched_tokens":16384}' \
  --multi_turn_scheduler pi_agent_scheduler_v2 \
  --max_turns 16 \
  --external_plugins "${V1_PLUGIN}" "${V2_PLUGIN}" "${SYNC_PLUGIN}" \
  --agent_template qwen3_5
