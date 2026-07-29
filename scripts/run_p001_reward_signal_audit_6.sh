#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <run-name>\n' "$0" >&2
  exit 2
fi

RUN_NAME="$1"
ROOT=/data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
R4_RUN_DIR="${ROOT}/runs/p001_crosshost_grpo_1step_20260728_r4"
R4_LORA_SHA256=224c2eb37844d6dbe8a260c7b72de6270f49691b1182ec050f83b210757a725e
CONTAINER="${PI_AGENT_ROLLOUT_CONTAINER:-llin-qwen36-grpo-pi-rollout-priv-host-0727}"
HTTP_PORT=28220
MAX_MODEL_LEN="${PI_AGENT_MAX_MODEL_LEN:-8192}"
COMPLETION_BUDGET="${PI_AGENT_COMPLETION_BUDGET:-2048}"
DATASET=/workspace/grpo_run/shared/datasets/train_20_unique_prompts.jsonl
VERIFIER_MANIFEST=/workspace/pi_rl/data/processed/strong_verified_27/verifier_manifest.jsonl
PLUGIN=/workspace/grpo_run/shared/pi_agent_grpo_plugin.py
AUDIT_SCRIPT=/workspace/grpo_run/shared/audit_online_grpo_reward_signal.py
LOAD_SCRIPT=/workspace/grpo_run/shared/load_shared_lora_adapter.py
PARTIAL_SUMMARY_SCRIPT=/workspace/grpo_run/shared/summarize_partial_reward_audit.py
ROLLOUT_INNER="${PI_AGENT_ROLLOUT_INNER:-/workspace/grpo_run/shared/run_qwen36_grpo_pi_rollout_inner.sh}"
REWARD_CONTRACT="${PI_AGENT_REWARD_CONTRACT:-v1}"
AUDIT_START_PROMPT="${PI_AGENT_AUDIT_START_PROMPT:-0}"
AUDIT_END_PROMPT="${PI_AGENT_AUDIT_END_PROMPT:-20}"
DEFER_SUMMARY="${PI_AGENT_DEFER_SUMMARY:-false}"
CONTAINER_RUN_DIR="/workspace/grpo_run/runs/${RUN_NAME}"
SYNC_DIR="${RUN_DIR}/shared_lora_sync"
SYNC_FILE="${SYNC_DIR}/adapter_flattened.pt"

case "${RUN_DIR}" in
  /data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727/runs/*) ;;
  *)
    printf 'refusing_run_dir=%s\n' "${RUN_DIR}" >&2
    exit 2
    ;;
esac
case "${CONTAINER}" in
  llin-*) ;;
  *)
    printf 'refusing_non_llin_container=%s\n' "${CONTAINER}" >&2
    exit 2
    ;;
esac
if [[ -e "${RUN_DIR}" ]]; then
  printf 'refusing_existing_run_dir=%s\n' "${RUN_DIR}" >&2
  exit 3
fi
test "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" = true
if docker top "${CONTAINER}" | grep -F 'swift.cli.rollout' | grep -v grep >/dev/null; then
  printf 'refusing_active_rollout_process=true\n' >&2
  exit 4
fi
test -r "${R4_RUN_DIR}/shared_lora_sync/adapter_flattened.pt"
test -r "${ROOT}/shared/$(basename "${ROLLOUT_INNER}")"
test -r "${ROOT}/shared/audit_online_grpo_reward_signal.py"
test -r "${ROOT}/shared/summarize_partial_reward_audit.py"
if [[ "${REWARD_CONTRACT}" != v1 && "${REWARD_CONTRACT}" != v2 ]]; then
  printf 'invalid_reward_contract=%s\n' "${REWARD_CONTRACT}" >&2
  exit 2
fi
if [[ "${DEFER_SUMMARY}" != true && "${DEFER_SUMMARY}" != false ]]; then
  printf 'invalid_defer_summary=%s\n' "${DEFER_SUMMARY}" >&2
  exit 2
fi
if ! [[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ && "${COMPLETION_BUDGET}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'invalid_token_budget max_model_len=%s completion_budget=%s\n' \
    "${MAX_MODEL_LEN}" "${COMPLETION_BUDGET}" >&2
  exit 2
fi
if (( COMPLETION_BUDGET >= MAX_MODEL_LEN )); then
  printf 'completion_budget_must_be_smaller_than_max_model_len=%s:%s\n' \
    "${COMPLETION_BUDGET}" "${MAX_MODEL_LEN}" >&2
  exit 2
fi
ACTUAL_R4_SHA256="$(sha256sum "${R4_RUN_DIR}/shared_lora_sync/adapter_flattened.pt" | awk '{print $1}')"
if [[ "${ACTUAL_R4_SHA256}" != "${R4_LORA_SHA256}" ]]; then
  printf 'r4_lora_sha256_mismatch expected=%s actual=%s\n' \
    "${R4_LORA_SHA256}" "${ACTUAL_R4_SHA256}" >&2
  exit 5
fi

mkdir -p "${SYNC_DIR}/versions" "${RUN_DIR}/audit"
cp --reflink=auto \
  "${R4_RUN_DIR}/shared_lora_sync/adapter_flattened.pt" \
  "${SYNC_DIR}/versions/r4-initial-${R4_LORA_SHA256}.pt"
ln -s "versions/r4-initial-${R4_LORA_SHA256}.pt" "${SYNC_FILE}"
printf '%s\n' "$(date -Is)" >"${RUN_DIR}/host_started_at"

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'host=%s\n' "$(hostname)"
  printf 'container=%s\n' "${CONTAINER}"
  printf 'policy_source_run=%s\n' "p001_crosshost_grpo_1step_20260728_r4"
  printf 'policy_lora_sha256=%s\n' "${R4_LORA_SHA256}"
  printf 'policy_update_performed=false\n'
  printf 'prompt_count=20\nsamples_per_prompt=8\ntrajectory_count=160\n'
  printf 'max_model_len=%s\ncompletion_budget=%s\n' \
    "${MAX_MODEL_LEN}" "${COMPLETION_BUDGET}"
  printf 'temperature=0.9\ntop_p=0.95\n'
  printf 'quality_claims_allowed=false\n'
  printf 'rollout_inner=%s\n' "${ROLLOUT_INNER}"
  printf 'reward_contract=%s\n' "${REWARD_CONTRACT}"
  printf 'audit_prompt_range=%s:%s\n' "${AUDIT_START_PROMPT}" "${AUDIT_END_PROMPT}"
  printf 'defer_summary=%s\n' "${DEFER_SUMMARY}"
  docker inspect -f \
    'image={{.Config.Image}} status={{.State.Status}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}}' \
    "${CONTAINER}"
  docker exec "${CONTAINER}" sha256sum \
    "${DATASET}" \
    "${VERIFIER_MANIFEST}" \
    "${PLUGIN}"
  sha256sum \
    "${ROOT}/shared/audit_online_grpo_reward_signal.py" \
    "${ROOT}/shared/load_shared_lora_adapter.py" \
    "${SYNC_FILE}"
} >"${RUN_DIR}/host_environment_summary.txt"

(
  while true; do
    printf 'timestamp=%s\n' "$(date -Is)"
    npu-smi info
    sleep 10
  done
) >"${RUN_DIR}/npu_smi_timeseries.log" 2>&1 &
SAMPLER_PID=$!
printf '%s\n' "${SAMPLER_PID}" >"${RUN_DIR}/sampler.pid"

cleanup() {
  set +e
  docker restart -t 10 "${CONTAINER}" >/dev/null
  kill -TERM "${SAMPLER_PID}" 2>/dev/null
  wait "${SAMPLER_PID}" 2>/dev/null
  npu-smi info >"${RUN_DIR}/npu_after.txt" 2>&1
  printf '%s\n' "$(date -Is)" >"${RUN_DIR}/host_completed_at"
}
trap cleanup EXIT

docker exec "${CONTAINER}" \
  bash "${ROLLOUT_INNER}" \
  "${RUN_NAME}" "${MAX_MODEL_LEN}" "${COMPLETION_BUDGET}" "${HTTP_PORT}" \
  >"${RUN_DIR}/rollout.log" 2>&1 &
ROLLOUT_WRAPPER_PID=$!
printf '%s\n' "${ROLLOUT_WRAPPER_PID}" >"${RUN_DIR}/rollout_wrapper.pid"

ROLLOUT_READY=false
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health/" >/dev/null 2>&1; then
    ROLLOUT_READY=true
    break
  fi
  if ! kill -0 "${ROLLOUT_WRAPPER_PID}" 2>/dev/null; then
    break
  fi
  sleep 5
done
printf '%s\n' "${ROLLOUT_READY}" >"${RUN_DIR}/rollout_ready"
if [[ "${ROLLOUT_READY}" != true ]]; then
  printf 'rollout_not_ready=true\n' >&2
  exit 6
fi

docker exec "${CONTAINER}" python "${LOAD_SCRIPT}" \
  "${CONTAINER_RUN_DIR}/shared_lora_sync/adapter_flattened.pt" \
  --base-url "http://127.0.0.1:${HTTP_PORT}" \
  --expected-sha256 "${R4_LORA_SHA256}" \
  >"${RUN_DIR}/adapter_load_summary.json"

set +e
AUDIT_EXTRA_ARGS=(
  --reward-contract "${REWARD_CONTRACT}"
  --start-prompt "${AUDIT_START_PROMPT}"
  --end-prompt "${AUDIT_END_PROMPT}"
)
if [[ "${DEFER_SUMMARY}" == true ]]; then
  AUDIT_EXTRA_ARGS+=(--defer-summary)
fi
docker exec \
  -e PI_AGENT_VERIFIER_MANIFEST="${VERIFIER_MANIFEST}" \
  -e PYTHONPATH=/workspace/grpo_run/shared/rollout_python_stubs:/workspace/ms-swift \
  "${CONTAINER}" python "${AUDIT_SCRIPT}" \
  "${DATASET}" "${VERIFIER_MANIFEST}" "${PLUGIN}" "${CONTAINER_RUN_DIR}/audit" \
  --base-url "http://127.0.0.1:${HTTP_PORT}" \
  --samples-per-prompt 8 \
  --max-tokens "${COMPLETION_BUDGET}" \
  --temperature 0.9 \
  --top-p 0.95 \
  --timeout 1800 \
  "${AUDIT_EXTRA_ARGS[@]}" \
  >"${RUN_DIR}/audit.log" 2>&1
AUDIT_EXIT=$?
set -e
printf '%s\n' "${AUDIT_EXIT}" >"${RUN_DIR}/audit_exit_code"
if [[ "${AUDIT_EXIT}" -ne 0 ]]; then
  exit "${AUDIT_EXIT}"
fi

if [[ "${DEFER_SUMMARY}" == true ]]; then
  docker exec "${CONTAINER}" python "${PARTIAL_SUMMARY_SCRIPT}" \
    "${CONTAINER_RUN_DIR}/audit/groups" \
    --output "${CONTAINER_RUN_DIR}/audit/partial_summary.json" \
    >"${RUN_DIR}/audit_safe_summary.json"
else
  python - "${RUN_DIR}/audit/summary.json" <<'PY' >"${RUN_DIR}/audit_safe_summary.json"
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
safe = {
    "schema": summary["schema"],
    "started_at": summary["started_at"],
    "completed_at": summary["completed_at"],
    "policy_update_performed": summary["policy_update_performed"],
    "sampling": summary["sampling"],
    "artifacts": summary["artifacts"],
    "aggregate": summary["aggregate"],
    "per_prompt": summary["per_prompt"],
    "decision_gate": summary["decision_gate"],
}
print(json.dumps(safe, ensure_ascii=False, indent=2))
PY
fi

printf 'audit_complete=true\n' >>"${RUN_DIR}/host_environment_summary.txt"
exit 0
