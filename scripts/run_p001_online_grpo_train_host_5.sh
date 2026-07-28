#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 7 ]]; then
  printf 'usage: %s <container> <run-name> <max-length> <completion-budget> <train-iters> <master-port> <group-port>\n' "$0" >&2
  exit 2
fi

CONTAINER="$1"
RUN_NAME="$2"
MAX_LENGTH="$3"
COMPLETION_BUDGET="$4"
TRAIN_ITERS="$5"
MASTER_PORT="$6"
GROUP_PORT="$7"
HOST_RUN_DIR="/data3/llin/qwen3.6-27b-trajory-dpo/online_grpo/runs/${RUN_NAME}"

case "${HOST_RUN_DIR}" in
  /data3/llin/qwen3.6-27b-trajory-dpo/online_grpo/runs/*) ;;
  *)
    printf 'refusing_host_run_dir=%s\n' "${HOST_RUN_DIR}" >&2
    exit 2
    ;;
esac

mkdir -p "${HOST_RUN_DIR}"
printf '%s\n' "$(date -Is)" >"${HOST_RUN_DIR}/training_host_started_at"

docker exec "${CONTAINER}" \
  bash /workspace/grpo_run/shared/run_p001_online_grpo_train_inner.sh \
  "${RUN_NAME}" "${MAX_LENGTH}" "${COMPLETION_BUDGET}" "${TRAIN_ITERS}" \
  "${MASTER_PORT}" 127.0.0.1 "${GROUP_PORT}"
RC=$?

printf '%s\n' "${RC}" >"${HOST_RUN_DIR}/training_exit_code"
printf '%s\n' "$(date -Is)" >"${HOST_RUN_DIR}/training_host_completed_at"
exit "${RC}"
