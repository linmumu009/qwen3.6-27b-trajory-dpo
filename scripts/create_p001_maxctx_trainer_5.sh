#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONTAINER=llin-qwen36-grpo-trainer-m05-p001-dpo-base
CONTAINER=llin-qwen36-grpo-trainer-m05-p001-maxctx-0729
IMAGE=llin-rl-dpo-p2-base:20260707
VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

case "${CONTAINER}" in
  llin-*) ;;
  *) exit 2 ;;
esac
case "${IMAGE}" in
  llin-*) ;;
  *) exit 2 ;;
esac
docker container inspect "${SOURCE_CONTAINER}" >/dev/null
docker image inspect "${IMAGE}" >/dev/null
if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  printf 'refusing_existing_container=%s\n' "${CONTAINER}" >&2
  exit 3
fi

docker run -d \
  --name "${CONTAINER}" \
  --privileged \
  --network host \
  --ipc host \
  --restart no \
  --volumes-from "${SOURCE_CONTAINER}" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e HCCL_CONNECT_TIMEOUT=1800 \
  -e ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -e ASCEND_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -e PI_AGENT_TRAIN_NPROC_PER_NODE=16 \
  -e PI_AGENT_TRAIN_TP_SIZE=8 \
  -e PI_AGENT_TRAIN_CP_SIZE=2 \
  -e PI_AGENT_TRAIN_PP_SIZE=1 \
  --entrypoint bash \
  "${IMAGE}" -lc 'exec sleep infinity'

docker inspect "${CONTAINER}" --format \
  'name={{.Name}} status={{.State.Status}} image={{.Config.Image}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}}'
docker exec "${CONTAINER}" bash -lc \
  'python -c '\''import torch, torch_npu; print(f"npu_count={torch.npu.device_count()} available={torch.npu.is_available()}")'\'''
