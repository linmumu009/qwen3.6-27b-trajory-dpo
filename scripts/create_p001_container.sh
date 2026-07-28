#!/usr/bin/env bash
set -euo pipefail

HOST_ROOT="${HOST_ROOT:-/data3/llin/qwen3.6-27b-trajory-dpo}"
IMAGE="${IMAGE:-llin-qwen36-p001-megatron:20260728}"
CONTAINER="${CONTAINER:-llin-qwen36-p001-megatron}"
MODEL_ROOT="${MODEL_ROOT:-/data/models/Qwen3.6-27B}"

test -d "${HOST_ROOT}/reference/ms-swift-padding-buckets"
test -d "${HOST_ROOT}/reference/Megatron-Core-pypi-0.16.0-chunked-ce"
test -r "${MODEL_ROOT}/config.json"
docker image inspect "${IMAGE}" >/dev/null

if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${CONTAINER}" >&2
  exit 3
fi

mount_args=(
  -v "${HOST_ROOT}:/workspace/p001"
  -v "${MODEL_ROOT}:/models/Qwen3.6-27B:ro"
  -v "/usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro"
  -v "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
  --device "/dev/davinci_manager:/dev/davinci_manager"
  --device "/dev/devmm_svm:/dev/devmm_svm"
  --device "/dev/hisi_hdc:/dev/hisi_hdc"
)
for device_index in $(seq 0 15); do
  mount_args+=(--device "/dev/davinci${device_index}:/dev/davinci${device_index}")
done
for bind_path in \
  /usr/local/sbin/npu-smi \
  /usr/local/bin/npu-smi \
  /usr/local/dcmi \
  /etc/ascend_install.info \
  /etc/hccn.conf
do
  if [[ -e "${bind_path}" ]]; then
    mount_args+=(-v "${bind_path}:${bind_path}:ro")
  fi
done

docker run -d \
  --name "${CONTAINER}" \
  --shm-size 64g \
  --ipc host \
  --network host \
  --workdir /workspace/p001 \
  -e HCCL_CONNECT_TIMEOUT=1800 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e PHYSICAL_NPU_DEVICE_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  "${mount_args[@]}" \
  "${IMAGE}" \
  bash -lc 'sleep infinity'
