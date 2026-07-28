#!/usr/bin/env bash
set -euo pipefail

SOURCE_HOST="${SOURCE_HOST:-root@192.168.202.4}"
IMAGE="${IMAGE:-llin-rl-grpo:pi-deps-20260727}"

test "$(hostname -I | tr ' ' '\n' | grep -Fx 192.168.202.5 | wc -l)" -ge 1
ssh -o BatchMode=yes "${SOURCE_HOST}" "docker image inspect '${IMAGE}' >/dev/null"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  printf 'image_already_present=%s\n' "${IMAGE}"
  docker image inspect "${IMAGE}" --format 'id={{.Id}} size={{.Size}}'
  exit 0
fi

printf 'transfer_started_at=%s\n' "$(date -Is)"
printf 'source_host=%s\nimage=%s\n' "${SOURCE_HOST}" "${IMAGE}"
ssh -o BatchMode=yes "${SOURCE_HOST}" "docker image save '${IMAGE}'" |
  docker image load
printf 'transfer_finished_at=%s\n' "$(date -Is)"
docker image inspect "${IMAGE}" --format 'id={{.Id}} size={{.Size}}'
