#!/usr/bin/env bash
set -euo pipefail

cd /workspace/p001

STAGE="${STAGE:?set STAGE to sft, continued_sft, dpo, rpo, or randomized_rpo}"
DATASET="${DATASET:?set DATASET to a container-visible JSONL path}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a container-visible output directory}"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
SAVE_STEPS="${SAVE_STEPS:-${TRAIN_ITERS}}"
SEED="${SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29671}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
REFERENCE_ADAPTER_PATH="${REFERENCE_ADAPTER_PATH:-${ADAPTER_PATH}}"
DATA_MANIFEST="${DATA_MANIFEST:-}"

case "${STAGE}" in
  sft)
    ENTRYPOINT="/workspace/p001/reference/ms-swift-padding-buckets/swift/cli/_megatron/sft.py"
    ;;
  continued_sft)
    ENTRYPOINT="/workspace/p001/reference/ms-swift-padding-buckets/swift/cli/_megatron/sft.py"
    test -n "${ADAPTER_PATH}"
    ;;
  dpo | rpo | randomized_rpo)
    ENTRYPOINT="/workspace/p001/reference/ms-swift-padding-buckets/swift/cli/_megatron/rlhf.py"
    test -n "${ADAPTER_PATH}"
    test -n "${REFERENCE_ADAPTER_PATH}"
    ;;
  *)
    printf 'unsupported STAGE=%s\n' "${STAGE}" >&2
    exit 2
    ;;
esac

MODEL="/models/Qwen3.6-27B"
test -r "${MODEL}/config.json"
test -r "${MODEL}/model.safetensors.index.json"
test -r "${DATASET}"
if [[ -n "${DATA_MANIFEST}" ]]; then
  test -r "${DATA_MANIFEST}"
fi

mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_p001_megatron_inner.sh"
if [[ -n "${DATA_MANIFEST}" ]]; then
  cp "${DATA_MANIFEST}" "${OUTPUT_DIR}/data_manifest.jsonl"
fi

export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export MCORE_CE_SEQ_CHUNK_SIZE=1024
export SWIFT_MEGATRON_PADDING_TO=1024
export PYTHONPATH="/workspace/p001/scripts/hccl_rank_ports:/workspace/p001/reference/msgspec:/workspace/p001/reference/mcore-bridge-1.6.0-cp2:/workspace/p001/reference/Megatron-Core-pypi-0.16.0-chunked-ce:/workspace/p001/reference/MindSpeed-core_r0.16.0:/workspace/p001/reference/ms-swift-padding-buckets:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EVENT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=7200
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=7200

{
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'stage=%s\n' "${STAGE}"
  printf 'model=%s\n' "${MODEL}"
  printf 'dataset=%s\n' "${DATASET}"
  printf 'data_manifest=%s\n' "${DATA_MANIFEST}"
  printf 'adapter_path=%s\n' "${ADAPTER_PATH}"
  printf 'reference_adapter_path=%s\n' "${REFERENCE_ADAPTER_PATH}"
  printf 'tp=8\npp=1\ncp=2\nsequence_parallel=true\n'
  printf 'max_length=%s\ntrain_iters=%s\n' "${MAX_LENGTH}" "${TRAIN_ITERS}"
  printf 'learning_rate=%s\nseed=%s\n' "${LEARNING_RATE}" "${SEED}"
  printf 'dataset_shuffle=false\ntrain_dataloader_shuffle=false\n'
  printf 'lora_rank=8\nlora_alpha=32\n'
  printf 'lora_target_modules=linear_qkv,linear_fc1\n'
  printf 'dpo_beta=0.1\n'
  if [[ "${STAGE}" == "rpo" || "${STAGE}" == "randomized_rpo" ]]; then
    printf 'rpo_alpha=1.0\n'
  fi
  python -V
  python -c 'import torch, torch_npu, transformers, swift; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"swift={swift.__version__}")'
  sha256sum "${MODEL}/config.json" "${MODEL}/model.safetensors.index.json" "${DATASET}"
  if [[ -n "${DATA_MANIFEST}" ]]; then
    sha256sum "${DATA_MANIFEST}"
  fi
} >"${OUTPUT_DIR}/environment_summary.txt" 2>&1

npu-smi info >"${OUTPUT_DIR}/npu_before.txt" 2>&1

common_args=(
  --model "${MODEL}"
  --dataset "${DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --tensor_model_parallel_size 8
  --pipeline_model_parallel_size 1
  --context_parallel_size 2
  --sequence_parallel true
  --tuner_type lora
  --target_modules linear_qkv linear_fc1
  --lora_rank 8
  --lora_alpha 32
  --lora_dropout 0
  --max_length "${MAX_LENGTH}"
  --truncation_strategy delete
  --padding_free false
  --micro_batch_size 1
  --global_batch_size 1
  --train_iters "${TRAIN_ITERS}"
  --lr "${LEARNING_RATE}"
  --lr_warmup_fraction 0.03
  --min_lr 1e-6
  --lr_decay_style cosine
  --weight_decay 0.0
  --adam_beta1 0.9
  --adam_beta2 0.95
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --use_distributed_optimizer false
  --attention_backend flash
  --cross_entropy_loss_fusion true
  --gradient_accumulation_fusion false
  --masked_softmax_fusion false
  --merge_lora false
  --save_steps "${SAVE_STEPS}"
  --logging_steps 1
  --dataloader_num_workers 0
  --dataloader_pin_memory false
  --dataset_num_proc 1
  --dataset_shuffle false
  --train_dataloader_shuffle false
  --seed "${SEED}"
)

if [[ -n "${ADAPTER_PATH}" ]]; then
  common_args+=(
    --mcore_adapter "${ADAPTER_PATH}"
    --no_load_optim true
    --no_load_rng true
    --finetune true
  )
fi

if [[ "${STAGE}" == "dpo" || "${STAGE}" == "rpo" || "${STAGE}" == "randomized_rpo" ]]; then
  common_args=(
    --rlhf_type dpo
    --beta 0.1
    --mcore_ref_adapter "${REFERENCE_ADAPTER_PATH}"
    "${common_args[@]}"
  )
fi

if [[ "${STAGE}" == "rpo" || "${STAGE}" == "randomized_rpo" ]]; then
  common_args+=(--rpo_alpha 1.0)
fi

set +e
python -m torch.distributed.run \
  --nproc_per_node 16 \
  --master_port "${MASTER_PORT}" \
  "${ENTRYPOINT}" \
  "${common_args[@]}"
exit_code=$?
set -e

printf '%s\n' "${exit_code}" >"${OUTPUT_DIR}/exit_code"
npu-smi info >"${OUTPUT_DIR}/npu_after.txt" 2>&1
printf 'finished_at=%s\n' "$(date -Is)" >>"${OUTPUT_DIR}/environment_summary.txt"
exit "${exit_code}"
