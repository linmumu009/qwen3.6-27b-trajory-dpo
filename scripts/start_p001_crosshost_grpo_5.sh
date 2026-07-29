#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <run-name>\n' "$0" >&2
  exit 2
fi

RUN_NAME="$1"
PROJECT_ROOT=/data3/llin/qwen3.6-27b-trajory-dpo
RUN_DIR="${PROJECT_ROOT}/online_grpo/runs/${RUN_NAME}"
SYNC_FILE="${RUN_DIR}/shared_lora_sync/adapter_flattened.pt"
REMOTE_HOST=root@192.168.202.4
REMOTE_RUN_DIR="/data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727/runs/${RUN_NAME}"
REMOTE_SYNC_DIR="${REMOTE_RUN_DIR}/shared_lora_sync"
CONTAINER="${PI_AGENT_TRAIN_CONTAINER:-llin-qwen36-grpo-trainer-m05-p001-dpo-base}"
MAX_LENGTH="${PI_AGENT_MAX_LENGTH:-4096}"
COMPLETION_BUDGET="${PI_AGENT_COMPLETION_BUDGET:-2048}"
TRAIN_ITERS="${PI_AGENT_TRAIN_ITERS:-1}"
MASTER_PORT="${PI_AGENT_MASTER_PORT:-29681}"
GROUP_PORT="${PI_AGENT_GROUP_PORT:-28221}"

case "${RUN_DIR}" in
  /data3/llin/qwen3.6-27b-trajory-dpo/online_grpo/runs/*) ;;
  *)
    printf 'refusing_run_dir=%s\n' "${RUN_DIR}" >&2
    exit 2
    ;;
esac

if [[ -e "${RUN_DIR}/control_started_at" ]]; then
  printf 'run_already_started=%s\n' "${RUN_DIR}" >&2
  exit 3
fi

test "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" = true
case "${CONTAINER}" in
  llin-*) ;;
  *)
    printf 'refusing_non_llin_container=%s\n' "${CONTAINER}" >&2
    exit 2
    ;;
esac
ssh -o BatchMode=yes "${REMOTE_HOST}" "test -d '${REMOTE_RUN_DIR}'"

mkdir -p "${RUN_DIR}/shared_lora_sync"
printf '%s\n' "$(date -Is)" >"${RUN_DIR}/control_started_at"

nohup ssh \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=4 \
  -N \
  -L 28220:127.0.0.1:28220 \
  -L 28221:127.0.0.1:28221 \
  "${REMOTE_HOST}" \
  >"${RUN_DIR}/ssh_tunnel.log" 2>&1 &
TUNNEL_PID=$!
printf '%s\n' "${TUNNEL_PID}" >"${RUN_DIR}/ssh_tunnel.pid"

HEALTHY=false
for _ in $(seq 1 30); do
  if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    printf 'ssh_tunnel_exited_early\n' >&2
    exit 4
  fi
  if [[ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:28220/health/ || true)" = 200 ]]; then
    HEALTHY=true
    break
  fi
  sleep 1
done
if [[ "${HEALTHY}" != true ]]; then
  printf 'rollout_health_timeout\n' >&2
  exit 5
fi

nohup bash "${PROJECT_ROOT}/scripts/watch_cross_host_lora_sync.sh" \
  "${SYNC_FILE}" "${REMOTE_HOST}" "${REMOTE_SYNC_DIR}" \
  >"${RUN_DIR}/lora_watcher.log" 2>&1 &
WATCHER_PID=$!
printf '%s\n' "${WATCHER_PID}" >"${RUN_DIR}/lora_watcher.pid"

nohup bash "${PROJECT_ROOT}/scripts/run_p001_online_grpo_train_host_5.sh" \
  "${CONTAINER}" "${RUN_NAME}" "${MAX_LENGTH}" "${COMPLETION_BUDGET}" \
  "${TRAIN_ITERS}" "${MASTER_PORT}" "${GROUP_PORT}" \
  >"${RUN_DIR}/training.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" >"${RUN_DIR}/training_host.pid"

sleep 2
kill -0 "${TUNNEL_PID}"
kill -0 "${WATCHER_PID}"
kill -0 "${TRAIN_PID}"
printf 'run_dir=%s\ntunnel_pid=%s\nwatcher_pid=%s\ntraining_pid=%s\n' \
  "${RUN_DIR}" "${TUNNEL_PID}" "${WATCHER_PID}" "${TRAIN_PID}"
