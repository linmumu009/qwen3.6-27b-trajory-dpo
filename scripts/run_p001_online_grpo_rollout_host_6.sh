#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s <container> <run-name> <max-model-len> <completion-budget>\n' "$0" >&2
  exit 2
fi

CONTAINER="$1"
RUN_NAME="$2"
MAX_MODEL_LEN="$3"
COMPLETION_BUDGET="$4"
ROOT=/data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
INNER=/workspace/grpo_run/shared/run_qwen36_grpo_pi_rollout_v2_inner.sh
HTTP_PORT=28220

case "${CONTAINER}" in
  llin-*) ;;
  *)
    printf 'refusing_non_llin_container=%s\n' "${CONTAINER}" >&2
    exit 2
    ;;
esac
case "${RUN_DIR}" in
  "${ROOT}"/runs/*) ;;
  *) exit 2 ;;
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

mkdir -p "${RUN_DIR}"
printf '%s\n' "$(date -Is)" >"${RUN_DIR}/rollout_host_started_at"

ROLLOUT_PID=
SAMPLER_PID=
cleanup() {
  set +e
  if [[ -n "${ROLLOUT_PID}" ]]; then
    kill -TERM "${ROLLOUT_PID}" 2>/dev/null
    wait "${ROLLOUT_PID}" 2>/dev/null
  fi
  docker restart -t 10 "${CONTAINER}" >/dev/null 2>&1
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null
    wait "${SAMPLER_PID}" 2>/dev/null
  fi
  npu-smi info >"${RUN_DIR}/npu_after.txt" 2>&1
  printf '%s\n' "$(date -Is)" >"${RUN_DIR}/rollout_host_completed_at"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

(
  while true; do
    printf 'timestamp=%s\n' "$(date -Is)"
    npu-smi info
    sleep 5
  done
) >"${RUN_DIR}/npu_smi_timeseries.log" 2>&1 &
SAMPLER_PID=$!
printf '%s\n' "${SAMPLER_PID}" >"${RUN_DIR}/sampler.pid"

docker exec "${CONTAINER}" bash "${INNER}" \
  "${RUN_NAME}" "${MAX_MODEL_LEN}" "${COMPLETION_BUDGET}" "${HTTP_PORT}" \
  >"${RUN_DIR}/rollout.log" 2>&1 &
ROLLOUT_PID=$!
printf '%s\n' "${ROLLOUT_PID}" >"${RUN_DIR}/rollout_wrapper.pid"

READY=false
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health/" >/dev/null 2>&1; then
    READY=true
    break
  fi
  if ! kill -0 "${ROLLOUT_PID}" 2>/dev/null; then
    break
  fi
  sleep 5
done
printf '%s\n' "${READY}" >"${RUN_DIR}/rollout_ready"
if [[ "${READY}" != true ]]; then
  wait "${ROLLOUT_PID}"
fi
printf '%s\n' "$(date -Is)" >"${RUN_DIR}/rollout_ready_at"
wait "${ROLLOUT_PID}"
