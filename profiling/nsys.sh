#!/usr/bin/env bash
# nsys.sh — Profile train with Nsight Systems.
#
# Produces a .nsys-rep file openable in Nsight Systems GUI.
# NVTX ranges show: rl_episode_reset / sort_dispatch / rl_episode_kernel.
#
# Usage:
#   ROM=pokered.gb ./profiling/nsys.sh [extra train args...]
#   ROM=pokered.gb GENS=3 AGENTS=512 ./profiling/nsys.sh
#
# Output: profiling/reports/nsys_<timestamp>.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN="$REPO_ROOT/train"

if [[ ! -f "$TRAIN" ]]; then
  echo "ERROR: '$TRAIN' not found. Run 'make train' first."
  exit 1
fi

ROM="${ROM:-pokered.gb}"
if [[ ! -f "$ROM" ]]; then
  echo "ERROR: ROM not found: $ROM"
  echo "  Set ROM=/path/to/pokered.gb or copy it here."
  exit 1
fi

GENS="${GENS:-3}"
AGENTS="${AGENTS:-2048}"
FRAMES="${FRAMES:-600}"

REPORT_DIR="$REPO_ROOT/profiling/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPORT_DIR/nsys_${TIMESTAMP}"

echo "=== Nsight Systems Profile ==="
echo "  ROM:     $ROM"
echo "  agents:  $AGENTS"
echo "  gens:    $GENS"
echo "  frames:  $FRAMES"
echo "  output:  $OUT.nsys-rep"
echo ""

nsys profile \
  --trace=cuda,nvtx \
  --stats=true \
  --output="$OUT" \
  --force-overwrite=true \
  "$TRAIN" -n "$AGENTS" -g "$GENS" -f "$FRAMES" "$@" "$ROM"

echo ""
echo "Profile written: $OUT.nsys-rep"
echo "Open in Nsight Systems GUI or run:"
echo "  nsys stats $OUT.nsys-rep"
