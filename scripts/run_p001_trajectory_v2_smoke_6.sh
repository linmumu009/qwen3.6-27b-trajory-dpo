#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <run-name>\n' "$0" >&2
  exit 2
fi

export PI_AGENT_ROLLOUT_INNER=/workspace/grpo_run/shared/run_qwen36_grpo_pi_rollout_v2_inner.sh
export PI_AGENT_REWARD_CONTRACT=v2
export PI_AGENT_AUDIT_START_PROMPT="${PI_AGENT_AUDIT_START_PROMPT:-0}"
export PI_AGENT_AUDIT_END_PROMPT="${PI_AGENT_AUDIT_END_PROMPT:-1}"
export PI_AGENT_DEFER_SUMMARY=true

exec bash /data3/llin/trajory_sft/runs/qwen36_27b_grpo_pi_agent_20260727/shared/run_p001_reward_signal_audit_6.sh "$1"
