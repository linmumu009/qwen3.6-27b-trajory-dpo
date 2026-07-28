#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-llin-qwen36-grpo-trainer-m05-p001}"
IMAGE="${IMAGE:-llin-rl-grpo:pi-deps-20260727}"
PROJECT_ROOT="${PROJECT_ROOT:-/data3/llin/qwen3.6-27b-trajory-dpo}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/online_grpo}"
MODEL_ROOT="${MODEL_ROOT:-/data/models/Qwen3.6-27B}"
TRAIN_REFERENCE_ROOT="${TRAIN_REFERENCE_ROOT:-${PROJECT_ROOT}}"

test "$(hostname -I | tr ' ' '\n' | grep -Fx 192.168.202.5 | wc -l)" -ge 1
docker image inspect "${IMAGE}" >/dev/null
case "${IMAGE}" in
  llin-*) ;;
  *)
    printf 'refusing_non_llin_image=%s\n' "${IMAGE}" >&2
    exit 4
    ;;
esac
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  printf 'refusing_existing_container=%s\n' "${CONTAINER_NAME}" >&2
  exit 3
fi

test -r "${MODEL_ROOT}/config.json"
test -d "${TRAIN_REFERENCE_ROOT}/reference/ms-swift-padding-buckets"
mkdir -p "${RUN_ROOT}/shared" "${RUN_ROOT}/runs"

container_command=(/bin/bash -lc 'exec sleep infinity')

docker run -d \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --restart no \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e HCCL_CONNECT_TIMEOUT=1800 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "${RUN_ROOT}:/workspace/grpo_run:rw" \
  -v "${MODEL_ROOT}:/models/Qwen3.6-27B:ro" \
  -v "${TRAIN_REFERENCE_ROOT}:/workspace/llin-rl-dpo:ro" \
  "${IMAGE}" \
  "${container_command[@]}"

docker inspect "${CONTAINER_NAME}" --format \
  'name={{.Name}} image={{.Config.Image}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}}'
docker exec "${CONTAINER_NAME}" bash -lc \
  'python -c '\''import torch, torch_npu; print(f"npu_count={torch.npu.device_count()} available={torch.npu.is_available()}")'\'''
