#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/amd_gpu_brief.sh [--watch SECONDS]

Shows only:
  GPU name
  GPU utilization
  VRAM used
  VRAM total

Examples:
  bash scripts/amd_gpu_brief.sh
  bash scripts/amd_gpu_brief.sh --watch 1
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

print_table() {
  local tmp_names tmp_mem tmp_use
  tmp_names="$(mktemp)"
  tmp_mem="$(mktemp)"
  tmp_use="$(mktemp)"

  cleanup() {
    rm -f "$tmp_names" "$tmp_mem" "$tmp_use"
  }
  trap cleanup RETURN

  rocm-smi --showproductname 2>/dev/null \
    | sed -n 's/^GPU\[\([0-9]\+\)\].*Card Series:[[:space:]]*\(.*\)$/\1\t\2/p' \
    | sed 's/[[:space:]]*$//' \
    | sort -n >"$tmp_names"

  amd-smi metric -m --json >"$tmp_mem"
  amd-smi metric -u --json >"$tmp_use"

  join -t $'\t' -1 1 -2 1 \
    <(jq -r '.[] | "\(.gpu)\t\(.mem_usage.used_vram.value)\t\(.mem_usage.total_vram.value)"' "$tmp_mem" | sort -n) \
    <(jq -r '.[] | "\(.gpu)\t\(.usage.gfx_activity.value)"' "$tmp_use" | sort -n) \
    | join -t $'\t' -1 1 -2 1 - "$tmp_names" \
    | awk -F '\t' '
        function gb(mb) { return sprintf("%.1f GB", mb / 1024) }
        BEGIN {
          print "GPU\tNAME\tUSE\tVRAM USED\tVRAM TOTAL"
          print "---\t----\t---\t---------\t----------"
        }
        {
          printf "%s\t%s\t%s%%\t%s\t%s\n", $1, $5, $4, gb($2), gb($3)
        }
      ' | column -t -s $'\t'
}

WATCH_INTERVAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_INTERVAL="${2:?missing value for --watch}"
      shift 2
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

need_cmd amd-smi
need_cmd jq
need_cmd column

if [[ -n "$WATCH_INTERVAL" ]]; then
  SCRIPT_PATH=$(realpath "$0")
  exec watch -n "$WATCH_INTERVAL" "$SCRIPT_PATH"
else
  print_table
fi
