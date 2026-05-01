# Profiling Guide

## Quick start

```bash
# 1. Build
make train

# 2. Measure throughput
ROM=pokered.gb ./profiling/benchmark.sh

# 3. Timeline profile (open result in Nsight Systems GUI)
ROM=pokered.gb ./profiling/nsys.sh

# 4. Deep kernel profile (open result in Nsight Compute GUI)
ROM=pokered.gb ./profiling/ncu.sh
```

## What each tool tells you

### `benchmark.sh` — throughput regression testing
- Measures state-frames/sec at 256/512/1024/2048 agents
- Run before and after a change; diff the outputs

### `nsys.sh` — timeline (where does time go?)
Produces `profiling/reports/nsys_*.nsys-rep`. Open in Nsight Systems GUI.

Key things to look at:
| Section | What it shows |
|---------|--------------|
| **CUDA API** | `cudaDeviceSynchronize` dominates = kernel is the bottleneck (good) |
| **NVTX ranges** | `rl_episode_kernel` vs `sort_dispatch` vs `rl_episode_reset` time split |
| **CUDA kernels** | Per-launch duration; gaps = CPU overhead or sync stalls |
| **Memory transfers** | H2D/D2H bandwidth; should be small vs kernel time |

### `ncu.sh` — per-kernel hardware counters
Produces `profiling/reports/ncu_*.ncu-rep`. Open in Nsight Compute GUI.

Key metrics to investigate:

| Metric | What it means | Red flag |
|--------|---------------|----------|
| `sm__throughput` | % of peak SM throughput used | < 50% = starving the GPU |
| `l1tex__t_sector_hit_rate` | L1 cache hit % | < 50% = too many cache misses |
| `l2cache__hit_rate` | L2 cache hit % | < 70% = hitting DRAM often |
| `dram__throughput` | DRAM bandwidth utilisation | > 80% = bandwidth-bound |
| `warps_issue_stalled_long_scoreboard` | Warp stalls on memory latency | > 30% = cache thrashing |
| `sm__warps_active` | Occupancy | < 50% = launch config issues |

### What to look for in this codebase

1. **Local memory spills** — if `gpu_exec_block_T<>` temp arrays spill to local
   memory, you'll see high `l1tex__data_pipe_lsu_wavefronts_g2s_ld_local`.
   Fix: reduce block complexity, or shrink the tmp array via IR optimizations.

2. **Warp divergence in IR switch** — the `switch(ins->op)` in `gpu_exec_block_T`
   can cause divergence. Look at `smsp__warps_issue_stalled_wait`.

3. **Hash table miss rate** — check `GBStateGPU::fallback_count` after a run.
   High fallback means blocks not in CFG; run `make traces` to generate JIT.

4. **Sorted dispatch gain** — compare `gpu_run_frames_sorted_kernel` vs
   `gpu_run_frames_kernel` durations (enable both in Makefile temporarily).

## JIT hot-trace workflow

Block execution profile → JIT traces → faster kernel:

```bash
# Step 1: collect block-hit profile (5 generations)
make profile ROM=pokered.gb PROFILE_GENS=5

# Step 2: generate hot_traces.cuh from profile
make traces ROM=pokered.gb

# Step 3: rebuild with JIT traces enabled
make train

# Step 4: re-benchmark
ROM=pokered.gb ./profiling/benchmark.sh
```
