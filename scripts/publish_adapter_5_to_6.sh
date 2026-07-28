#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <checkpoint-dir> <artifact-name>\n' "$0" >&2
  exit 2
fi

SOURCE_DIR="$(readlink -f "$1")"
ARTIFACT_NAME="$2"
REMOTE_HOST="${REMOTE_HOST:-root@192.168.202.4}"
LOCAL_HANDOFF_ROOT="${LOCAL_HANDOFF_ROOT:-/data3/llin/qwen3.6-27b-trajory-dpo/handoff}"
REMOTE_HANDOFF_ROOT="${REMOTE_HANDOFF_ROOT:-/data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727/shared/p001_multihost_handoff}"
LOCAL_STAGE="${LOCAL_HANDOFF_ROOT}/${ARTIFACT_NAME}"
REMOTE_INCOMING="${REMOTE_HANDOFF_ROOT}/.incoming/${ARTIFACT_NAME}"
REMOTE_FINAL="${REMOTE_HANDOFF_ROOT}/adapters/${ARTIFACT_NAME}"

if [[ ! "${ARTIFACT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'artifact-name contains unsupported characters: %s\n' "${ARTIFACT_NAME}" >&2
  exit 2
fi

for name in adapter_config.json adapter_model.safetensors additional_config.json args.json; do
  test -r "${SOURCE_DIR}/${name}"
done
test ! -e "${LOCAL_STAGE}"
mkdir -p "${LOCAL_STAGE}/payload"

cp \
  "${SOURCE_DIR}/adapter_config.json" \
  "${SOURCE_DIR}/adapter_model.safetensors" \
  "${SOURCE_DIR}/additional_config.json" \
  "${SOURCE_DIR}/args.json" \
  "${LOCAL_STAGE}/payload/"

(
  cd "${LOCAL_STAGE}/payload"
  sha256sum \
    adapter_config.json \
    adapter_model.safetensors \
    additional_config.json \
    args.json >../source_files.sha256
  tar -cf ../adapter_payload.tar \
    adapter_config.json \
    adapter_model.safetensors \
    additional_config.json \
    args.json
)
(
  cd "${LOCAL_STAGE}"
  sha256sum adapter_payload.tar >adapter_payload.tar.sha256
)

ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "set -eu; test ! -e '${REMOTE_INCOMING}'; test ! -e '${REMOTE_FINAL}'; mkdir -p '${REMOTE_INCOMING}'"
rsync -a --info=stats2 \
  "${LOCAL_STAGE}/adapter_payload.tar" \
  "${LOCAL_STAGE}/adapter_payload.tar.sha256" \
  "${LOCAL_STAGE}/source_files.sha256" \
  "${REMOTE_HOST}:${REMOTE_INCOMING}/"
ssh -o BatchMode=yes "${REMOTE_HOST}" \
  "set -eu; cd '${REMOTE_INCOMING}'; sha256sum -c adapter_payload.tar.sha256; mkdir extracted; tar -xf adapter_payload.tar -C extracted; cd extracted; sha256sum -c ../source_files.sha256; mkdir -p '${REMOTE_HANDOFF_ROOT}/adapters'; mv '${REMOTE_INCOMING}/extracted' '${REMOTE_FINAL}'; printf 'published=%s\n' '${REMOTE_FINAL}'"
