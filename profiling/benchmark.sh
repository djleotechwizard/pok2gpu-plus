#!/usr/bin/env bash
# benchmark.sh — Measure training throughput (state-frames/sec).
#
# Runs several configurations and prints a comparison table.
# Useful for before/after comparisons when optimizing the GPU kernel.
#
# Usage:
#   ROM=pokered.gb ./profiling/benchmark.sh
#   ROM=pokered.gb GENS=10 ./profiling/benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN="$REPO_ROOT/rl_train"

if [[ ! -f "$TRAIN" ]]; then
  echo "ERROR: '$TRAIN' not found. Run 'make train' first."
  exit 1
fi

ROM="${ROM:-pokered.gb}"
if [[ ! -f "$ROM" ]]; then
  echo "ERROR: ROM not found: $ROM"
  exit 1
fi

GENS="${GENS:-5}"
FRAMES="${FRAMES:-600}"

echo "=== pok2gpu-plus Throughput Benchmark ==="
echo "  ROM:    $ROM"
echo "  gens:   $GENS per config"
echo "  frames: $FRAMES per episode"
echo ""
printf "%-12s %-10s %-20s\n" "agents" "ep_time(s)" "state-frames/sec"
printf "%-12s %-10s %-20s\n" "------" "----------" "----------------"

for AGENTS in 256 512 1024 2048; do
  # Capture stderr (where stats are printed), extract ep time
  OUTPUT=$("$TRAIN" -n "$AGENTS" -g "$GENS" -f "$FRAMES" -w 0 "$ROM" 2>&1 || true)
  # Parse the last "ep=X.Xs  Yk sf/s" line
  EP_TIME=$(echo "$OUTPUT" | grep -oP 'ep=\K[\d.]+' | tail -1)
  SFPS=$(echo "$OUTPUT"    | grep -oP '[\d.]+(?=k sf/s)' | tail -1)
  if [[ -n "$SFPS" ]]; then
    SFPS_FULL=$(echo "$SFPS * 1000" | bc 2>/dev/null || echo "${SFPS}000")
    printf "%-12s %-10s %-20s\n" "$AGENTS" "${EP_TIME:-?}" "${SFPS_FULL:-?}"
  else
    printf "%-12s %-10s %-20s\n" "$AGENTS" "?" "?"
  fi
done

echo ""
echo "Tip: compare before/after a code change:"
echo "  git stash && make train && ROM=$ROM GENS=10 ./profiling/benchmark.sh > before.txt"
echo "  git stash pop && make train && ROM=$ROM GENS=10 ./profiling/benchmark.sh > after.txt"
echo "  diff before.txt after.txt"
