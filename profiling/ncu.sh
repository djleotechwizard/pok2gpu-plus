#!/usr/bin/env bash
# ncu.sh — Deep per-kernel profiling with Nsight Compute.
#
# Collects the most useful metrics for diagnosing:
#   • warp stalls (memory latency, instruction fetch, sync)
#   • memory throughput vs peak (L1, L2, DRAM)
#   • theoretical vs achieved occupancy
#   • instruction mix (ALU vs LSU vs control)
#
# Usage:
#   ROM=pokered.gb ./profiling/ncu.sh [--set=full|default]
#
# Output: profiling/reports/ncu_<timestamp>.ncu-rep  (open in Nsight Compute GUI)
#         profiling/reports/ncu_<timestamp>.txt       (CLI summary)

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
  exit 1
fi

REPORT_DIR="$REPO_ROOT/profiling/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPORT_DIR/ncu_${TIMESTAMP}"

# Use 64 agents for 1 generation: small enough to finish quickly,
# large enough to fill a few warps.
AGENTS="${AGENTS:-64}"

# Metric set: "default" is fast; "full" is thorough but slow.
NCU_SET="${1:---set=default}"

echo "=== Nsight Compute Deep Dive ==="
echo "  kernel:  rl_episode_kernel"
echo "  agents:  $AGENTS  (1 generation)"
echo "  metrics: $NCU_SET"
echo "  output:  $OUT.ncu-rep"
echo ""

ncu \
  $NCU_SET \
  --kernel-name "rl_episode_kernel" \
  --launch-count 1 \
  --export "$OUT" \
  --force-overwrite \
  "$TRAIN" -n "$AGENTS" -g 1 -f 300 "$ROM"

echo ""
echo "Profile written: $OUT.ncu-rep"
echo "Open in Nsight Compute GUI, or run:"
echo "  ncu --import $OUT.ncu-rep --page details"
echo ""
echo "Quick CLI summary:"
ncu --import "$OUT.ncu-rep" \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
l2cache__hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warps_issue_stalled_wait_per_warp_active.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__sass_thread_inst_executed_op_fp32_pred_on.sum \
  2>/dev/null || true

echo ""
echo "To profile a specific kernel launch count, set AGENTS= and pass --launch-count."
