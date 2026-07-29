#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONTAINER=llin-qwen36-grpo-pi-rollout-priv-host-0727
CONTAINER=llin-qwen36-grpo-pi-rollout-m06-maxctx-0729
IMAGE=llin-vllm-ascend:grpo-pi-deps-20260727
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
  -e HCCL_CONNECT_TIMEOUT=1800 \
  -e HCCL_IF_BASE_PORT=27600 \
  -e ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -e ASCEND_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -e PYTHONPATH=/workspace/ms-swift \
  -e PI_AGENT_ROLLOUT_TP_SIZE=8 \
  -e PI_AGENT_ROLLOUT_DP_SIZE=2 \
  -e PI_AGENT_POLICY_TOKEN_RESERVE=16384 \
  -e PI_AGENT_OBSERVATION_TOKEN_LIMIT=8192 \
  -e PI_AGENT_PER_TOOL_OBSERVATION_LIMIT=2048 \
  -e PI_AGENT_FINALIZATION_TOKEN_RESERVE=2048 \
  -e PI_AGENT_PER_TURN_POLICY_TOKEN_LIMIT=4096 \
  --entrypoint bash \
  "${IMAGE}" -lc 'exec sleep infinity'

docker inspect "${CONTAINER}" --format \
  'name={{.Name}} status={{.State.Status}} image={{.Config.Image}} privileged={{.HostConfig.Privileged}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}}'
docker exec "${CONTAINER}" bash -lc \
  'python -c '\''import torch; print(f"npu_count={torch.npu.device_count()} available={torch.npu.is_available()}")'\'''
