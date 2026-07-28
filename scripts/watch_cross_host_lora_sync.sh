#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s <local-sync-file> <remote-host> <remote-sync-dir>\n' "$0" >&2
  exit 2
fi

LOCAL_SYNC_FILE="$(readlink -m "$1")"
REMOTE_HOST="$2"
REMOTE_SYNC_DIR="$3"
LOCAL_REQUEST="${LOCAL_SYNC_FILE%.pt}.request.json"
LOCAL_ACK="${LOCAL_SYNC_FILE%.pt}.ack.json"
EVENT_LOG="${LOCAL_SYNC_FILE%.pt}.host_watcher.jsonl"
LAST_TRANSFER_ID=""

case "${LOCAL_SYNC_FILE}" in
  /data3/llin/qwen3.6-27b-trajory-dpo/online_grpo/runs/*/shared_lora_sync/adapter_flattened.pt) ;;
  *)
    printf 'refusing_local_sync_path=%s\n' "${LOCAL_SYNC_FILE}" >&2
    exit 3
    ;;
esac
case "${REMOTE_SYNC_DIR}" in
  /data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727/runs/*/shared_lora_sync) ;;
  *)
    printf 'refusing_remote_sync_dir=%s\n' "${REMOTE_SYNC_DIR}" >&2
    exit 4
    ;;
esac

mkdir -p "$(dirname "${LOCAL_SYNC_FILE}")"
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "set -eu; mkdir -p '${REMOTE_SYNC_DIR}/.incoming' '${REMOTE_SYNC_DIR}/versions'"

printf '{"event":"watcher_started","time":"%s","remote_host":"%s"}\n' \
  "$(date -Is)" "${REMOTE_HOST}" >>"${EVENT_LOG}"

while true; do
  if [[ ! -s "${LOCAL_REQUEST}" ]]; then
    sleep 0.1
    continue
  fi
  read -r TRANSFER_ID EXPECTED_SHA EXPECTED_BYTES < <(
    python3 -c '
import json
import re
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
transfer_id = str(data["transfer_id"])
sha = str(data["sha256"])
size = int(data["bytes"])
if not re.fullmatch(r"[0-9]+-[0-9]+", transfer_id):
    raise SystemExit("invalid transfer_id")
if not re.fullmatch(r"[0-9a-f]{64}", sha):
    raise SystemExit("invalid sha256")
if size <= 0:
    raise SystemExit("invalid byte count")
print(transfer_id, sha, size)
' "${LOCAL_REQUEST}"
  )
  if [[ "${TRANSFER_ID}" == "${LAST_TRANSFER_ID}" ]]; then
    sleep 0.1
    continue
  fi

  ACTUAL_SHA="$(sha256sum "${LOCAL_SYNC_FILE}" | awk '{print $1}')"
  ACTUAL_BYTES="$(stat -c %s "${LOCAL_SYNC_FILE}")"
  test "${ACTUAL_SHA}" = "${EXPECTED_SHA}"
  test "${ACTUAL_BYTES}" = "${EXPECTED_BYTES}"

  VERSION_NAME="adapter-${TRANSFER_ID}-${EXPECTED_SHA}.pt"
  REMOTE_INCOMING="${REMOTE_SYNC_DIR}/.incoming/${VERSION_NAME}"
  REMOTE_VERSION="${REMOTE_SYNC_DIR}/versions/${VERSION_NAME}"
  rsync -a --partial "${LOCAL_SYNC_FILE}" "${REMOTE_HOST}:${REMOTE_INCOMING}"
  ssh -o BatchMode=yes "${REMOTE_HOST}" \
    "set -eu; test \"\$(sha256sum '${REMOTE_INCOMING}' | awk '{print \$1}')\" = '${EXPECTED_SHA}'; test \"\$(stat -c %s '${REMOTE_INCOMING}')\" = '${EXPECTED_BYTES}'; mv '${REMOTE_INCOMING}' '${REMOTE_VERSION}'; ln -s 'versions/${VERSION_NAME}' '${REMOTE_SYNC_DIR}/.adapter_flattened.pt.link-${TRANSFER_ID}'; mv -Tf '${REMOTE_SYNC_DIR}/.adapter_flattened.pt.link-${TRANSFER_ID}' '${REMOTE_SYNC_DIR}/adapter_flattened.pt'"

  ACK_TMP="${LOCAL_ACK}.tmp-$$"
  printf '{"format":"llin-cross-host-lora-sync-ack-v1","transfer_id":"%s","sha256":"%s","bytes":%s,"remote_version":"%s","acknowledged_at":"%s"}\n' \
    "${TRANSFER_ID}" "${EXPECTED_SHA}" "${EXPECTED_BYTES}" \
    "${REMOTE_VERSION}" "$(date -Is)" >"${ACK_TMP}"
  mv "${ACK_TMP}" "${LOCAL_ACK}"
  printf '{"event":"transfer_published","time":"%s","transfer_id":"%s","sha256":"%s","bytes":%s,"remote_version":"%s"}\n' \
    "$(date -Is)" "${TRANSFER_ID}" "${EXPECTED_SHA}" \
    "${EXPECTED_BYTES}" "${REMOTE_VERSION}" >>"${EVENT_LOG}"
  LAST_TRANSFER_ID="${TRANSFER_ID}"
done
