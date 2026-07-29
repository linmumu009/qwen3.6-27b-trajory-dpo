#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  printf 'usage: %s <run-name> <max-length> <completion-budget> <train-iters> <master-port> <rollout-ip> <group-port>\n' "$0" >&2
  exit 2
fi

RUN_NAME="$1"
MAX_LENGTH="$2"
COMPLETION_BUDGET="$3"
TRAIN_ITERS="$4"
MASTER_PORT="$5"
ROLLOUT_IP="$6"
GROUP_PORT="$7"

RUN_DIR="/workspace/grpo_run/runs/${RUN_NAME}"
DATASET="${PI_AGENT_TRAIN_DATASET:-/workspace/grpo_run/shared/train_20_unique_prompts.jsonl}"
MANIFEST=/workspace/grpo_run/shared/manifest.json
PLUGIN=/workspace/grpo_run/shared/pi_agent_grpo_plugin.py
V2_PLUGIN=/workspace/grpo_run/shared/pi_agent_grpo_v2_plugin.py
TRAJECTORY_CONTRACT=/workspace/grpo_run/shared/pi_trajectory_contract.py
SHARED_SYNC_PLUGIN=/workspace/grpo_run/shared/shared_file_lora_sync_patch.py
CROSS_HOST_PLUGIN=/workspace/grpo_run/shared/cross_host_lora_sync_patch.py
SERVER_MODE_PLUGIN=/workspace/grpo_run/shared/server_mode_no_local_vllm_patch.py
MODEL=/models/Qwen3.6-27B

test -r "${MODEL}/config.json"
test -r "${MODEL}/model.safetensors.index.json"
test -r "${DATASET}"
test -r "${MANIFEST}"
test -r "${PLUGIN}"
test -r "${V2_PLUGIN}"
test -r "${TRAJECTORY_CONTRACT}"
test -r "${SHARED_SYNC_PLUGIN}"
test -r "${CROSS_HOST_PLUGIN}"
test -r "${SERVER_MODE_PLUGIN}"
test -r /workspace/grpo_run/shared/verifier_manifest.jsonl
mkdir -p "${RUN_DIR}"

export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/Ascend/cann-9.0.0/bin:${PATH}"
export MCORE_CE_SEQ_CHUNK_SIZE=1024
export SWIFT_MEGATRON_PADDING_TO=1024
export PYTHONPATH="/workspace/llin-rl-dpo/scripts/hccl_rank_ports:/workspace/llin-rl-dpo/reference/msgspec:/workspace/llin-rl-dpo/reference/mcore-bridge-1.6.0-cp2:/workspace/llin-rl-dpo/reference/Megatron-Core-pypi-0.16.0-chunked-ce:/workspace/llin-rl-dpo/reference/MindSpeed-core_r0.16.0:/workspace/llin-rl-dpo/reference/ms-swift-padding-buckets:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EVENT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=7200
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PI_AGENT_VERIFIER_MANIFEST=/workspace/grpo_run/shared/verifier_manifest.jsonl
export LLIN_SHARED_LORA_SYNC_FILE="${RUN_DIR}/shared_lora_sync/adapter_flattened.pt"
export LLIN_SHARED_LORA_SYNC_ROLE=train
export LLIN_CROSS_HOST_SYNC_TIMEOUT=300
export LLIN_REMOTE_VLLM_VERSION=0.23.0
TRAIN_NPROC_PER_NODE="${PI_AGENT_TRAIN_NPROC_PER_NODE:-8}"
TRAIN_TP_SIZE="${PI_AGENT_TRAIN_TP_SIZE:-4}"
TRAIN_CP_SIZE="${PI_AGENT_TRAIN_CP_SIZE:-2}"
TRAIN_PP_SIZE="${PI_AGENT_TRAIN_PP_SIZE:-1}"
GENERATION_BATCH_SIZE="${PI_AGENT_GENERATION_BATCH_SIZE:-8}"
NUM_GENERATIONS="${PI_AGENT_NUM_GENERATIONS:-8}"
GLOBAL_BATCH_SIZE="${PI_AGENT_GLOBAL_BATCH_SIZE:-8}"
if (( TRAIN_TP_SIZE * TRAIN_CP_SIZE * TRAIN_PP_SIZE != TRAIN_NPROC_PER_NODE )); then
  printf 'train_topology_world_size_mismatch nproc=%s tp=%s cp=%s pp=%s\n' \
    "${TRAIN_NPROC_PER_NODE}" "${TRAIN_TP_SIZE}" \
    "${TRAIN_CP_SIZE}" "${TRAIN_PP_SIZE}" >&2
  exit 3
fi
if (( GENERATION_BATCH_SIZE < TRAIN_NPROC_PER_NODE )); then
  printf 'generation_batch_smaller_than_world_size batch=%s world=%s\n' \
    "${GENERATION_BATCH_SIZE}" "${TRAIN_NPROC_PER_NODE}" >&2
  exit 3
fi
if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  printf 'generation_batch_not_divisible batch=%s generations=%s\n' \
    "${GENERATION_BATCH_SIZE}" "${NUM_GENERATIONS}" >&2
  exit 3
fi
if (( GENERATION_BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
  printf 'generation_global_batch_mismatch generation=%s global=%s steps=1\n' \
    "${GENERATION_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" >&2
  exit 3
fi

{
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'model=%s\n' "${MODEL}"
  printf 'dataset=%s\n' "${DATASET}"
  printf 'max_length=%s\ncompletion_budget=%s\n' "${MAX_LENGTH}" "${COMPLETION_BUDGET}"
  printf 'train_iters=%s\n' "${TRAIN_ITERS}"
  printf 'train_topology=tp%s_pp%s_cp%s_sp\n' \
    "${TRAIN_TP_SIZE}" "${TRAIN_PP_SIZE}" "${TRAIN_CP_SIZE}"
  printf 'rollout_ip=%s\nrollout_http_port=28220\nrollout_group_port=%s\n' "${ROLLOUT_IP}" "${GROUP_PORT}"
  printf 'num_generations=%s\ngeneration_batch_size=%s\n' \
    "${NUM_GENERATIONS}" "${GENERATION_BATCH_SIZE}"
  printf 'global_batch_size=%s\n' "${GLOBAL_BATCH_SIZE}"
  printf 'weight_transport=versioned_rsync_atomic_symlink\n'
  printf 'lora_target_modules=linear_qkv,linear_fc1\n'
  printf 'trajectory_contract=llin-pi-trajectory-grpo-v2\n'
  printf 'loss_contract=assistant_and_tool_call_only\n'
  printf 'reward_contract=sparse_terminal_outcome_plus_verified_progress\n'
  printf 'quality_claims_allowed=false\n'
  python -V
  python -c 'import torch, torch_npu, transformers, swift; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"swift={swift.__version__}")'
  sha256sum \
    "${MODEL}/config.json" \
    "${MODEL}/model.safetensors.index.json" \
    "${DATASET}" \
    "${MANIFEST}" \
    "${PLUGIN}" \
    "${V2_PLUGIN}" \
    "${TRAJECTORY_CONTRACT}" \
    "${SHARED_SYNC_PLUGIN}" \
    "${CROSS_HOST_PLUGIN}" \
    "${SERVER_MODE_PLUGIN}"
} >"${RUN_DIR}/training_environment_summary.txt" 2>&1

python -m torch.distributed.run \
  --nproc_per_node "${TRAIN_NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  /workspace/llin-rl-dpo/reference/ms-swift-padding-buckets/swift/cli/_megatron/rlhf.py \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --output_dir "${RUN_DIR}/output" \
  --tensor_model_parallel_size "${TRAIN_TP_SIZE}" \
  --pipeline_model_parallel_size "${TRAIN_PP_SIZE}" \
  --context_parallel_size "${TRAIN_CP_SIZE}" \
  --sequence_parallel true \
  --tuner_type lora \
  --target_modules linear_qkv linear_fc1 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "${ROLLOUT_IP}" \
  --vllm_server_port 28220 \
  --vllm_server_group_port "${GROUP_PORT}" \
  --vllm_server_timeout 1800 \
  --vllm_enable_lora true \
  --vllm_server_pass_dataset true \
  --external_plugins "${PLUGIN}" "${V2_PLUGIN}" "${SERVER_MODE_PLUGIN}" "${SHARED_SYNC_PLUGIN}" "${CROSS_HOST_PLUGIN}" \
  --multi_turn_scheduler pi_agent_scheduler_v2 \
  --max_turns 16 \
  --reward_funcs pi_agent_trajectory_v2 \
  --loss_scale default \
  --num_generations "${NUM_GENERATIONS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --steps_per_generation 1 \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${COMPLETION_BUDGET}" \
  --completion_length_limit_scope total \
  --truncation_strategy delete \
  --padding_free false \
  --micro_batch_size 1 \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train_iters "${TRAIN_ITERS}" \
  --beta 0.04 \
  --temperature 0.9 \
  --top_p 0.95 \
  --lr 1e-5 \
  --lr_warmup_fraction 0.03 \
  --min_lr 1e-6 \
  --lr_decay_style cosine \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --use_distributed_optimizer false \
  --attention_backend flash \
  --cross_entropy_loss_fusion true \
  --gradient_accumulation_fusion false \
  --masked_softmax_fusion false \
  --merge_lora false \
  --save_steps "${TRAIN_ITERS}" \
  --logging_steps 1 \
  --log_completions true \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory false \
  --dataset_num_proc 1 \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --seed 42
