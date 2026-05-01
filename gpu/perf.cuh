#pragma once
/*
 * perf.cuh — CUDA performance instrumentation for pok2gpu-plus.
 *
 * Two complementary systems:
 *
 * 1. CUDA Event timers (host-side wall-clock per kernel launch)
 *    Usage:
 *      PERF_EVENT_DECLARE(sort_t);
 *      PERF_EVENT_START(sort_t);
 *      // ... kernel calls ...
 *      float ms = PERF_EVENT_STOP_MS(sort_t);
 *
 * 2. GpuPerfCounters — device-side per-state metrics accumulated
 *    during a batch run, then reduced to host for analysis.
 *    Enable with: gpu_perf_counters_enable(batch)
 *    Report with: gpu_perf_counters_report(batch)
 *
 * Environment control:
 *   POKE_PERF=1   enable per-episode timing breakdown
 *   POKE_PERF=2   also enable per-kernel hardware events (ncu mode)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

/* ── CUDA error-check helper ─────────────────────────────── */

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            exit(1);                                                  \
        }                                                             \
    } while (0)

/* ── CUDA Event timer macros ─────────────────────────────── */

#define PERF_EVENT_DECLARE(name) \
    cudaEvent_t _perf_##name##_start, _perf_##name##_stop

#define PERF_EVENT_CREATE(name) \
    do { \
        CUDA_CHECK(cudaEventCreate(&_perf_##name##_start)); \
        CUDA_CHECK(cudaEventCreate(&_perf_##name##_stop));  \
    } while (0)

#define PERF_EVENT_DESTROY(name) \
    do { \
        cudaEventDestroy(_perf_##name##_start); \
        cudaEventDestroy(_perf_##name##_stop);  \
    } while (0)

#define PERF_EVENT_START(name) \
    CUDA_CHECK(cudaEventRecord(_perf_##name##_start))

#define PERF_EVENT_STOP_MS(name, out_ms) \
    do { \
        CUDA_CHECK(cudaEventRecord(_perf_##name##_stop)); \
        CUDA_CHECK(cudaEventSynchronize(_perf_##name##_stop)); \
        CUDA_CHECK(cudaEventElapsedTime(&(out_ms), \
            _perf_##name##_start, _perf_##name##_stop)); \
    } while (0)

/* ── Per-episode timing breakdown ────────────────────────── */

typedef struct {
    float sort_ms;       /* CUB radix sort (sorted dispatch) */
    float kernel_ms;     /* rl_episode_kernel execution */
    float h2d_ms;        /* host-to-device transfers */
    float d2h_ms;        /* device-to-host transfers */
    float total_ms;
    /* Derived */
    double state_frames_per_sec;
    double gb_fps_per_agent;     /* simulated GB fps per individual agent */
} EpisodeTiming;

static inline void episode_timing_print(const EpisodeTiming *t, int n_states, int ep_frames) {
    fprintf(stderr,
        "[perf]  sort=%.2fms  kernel=%.2fms  h2d=%.2fms  d2h=%.2fms  total=%.2fms\n"
        "[perf]  throughput: %.1fk state-frames/sec  (%.1f sim-fps per agent)\n",
        t->sort_ms, t->kernel_ms, t->h2d_ms, t->d2h_ms, t->total_ms,
        t->state_frames_per_sec / 1000.0,
        t->gb_fps_per_agent);
    (void)n_states; (void)ep_frames;
}

/* ── Per-state device counters ───────────────────────────── */
/*
 * Accumulated during gpu_ir_step() calls and reduced to host.
 * Each GBStateGPU already has fallback_cycles / fallback_count
 * as built-in counters.  These extended counters are optional
 * and stored in a separate parallel array to keep GBStateGPU small.
 */

typedef struct {
    uint32_t ir_blocks_exec;     /* blocks dispatched via IR interpreter */
    uint32_t trace_hits;         /* blocks dispatched via JIT hot traces  */
    uint32_t interrupt_services; /* interrupt service routine calls       */
    uint32_t halt_skips;         /* HALT fast-forward events              */
    uint64_t ir_cycles;          /* T-cycles consumed via IR path         */
    uint64_t fallback_cycles;    /* T-cycles consumed via SM83 fallback   */
} GpuPerfCounters;

/* ── Aggregate statistics (mean across all states) ───────── */

typedef struct {
    double ir_block_rate;    /* IR blocks/frame */
    double trace_hit_pct;    /* JIT trace hit % of total dispatches */
    double fallback_pct;     /* fallback % of total T-cycles */
    double interrupt_rate;   /* interrupts/frame */
    double halt_skip_rate;   /* halt fast-forwards/frame */
} GpuPerfSummary;

static inline void gpu_perf_summary_print(const GpuPerfSummary *s) {
    fprintf(stderr,
        "[perf]  IR blocks/frame=%.1f  JIT hit=%.1f%%  fallback=%.2f%%  "
        "interrupts/frame=%.2f  halt-skips/frame=%.2f\n",
        s->ir_block_rate,
        s->trace_hit_pct,
        s->fallback_pct,
        s->interrupt_rate,
        s->halt_skip_rate);
}

/* ── nsys / ncu annotation helpers ──────────────────────── */
/*
 * When running under Nsight Systems (nsys profile ./train ...),
 * these push/pop NVTX ranges that appear as coloured bands in the
 * timeline, making it easy to identify sort vs. kernel vs. transfers.
 *
 * Link with -lnvToolsExt if NVTX is available, otherwise no-ops.
 */

#ifdef POKE_NVTX
#  include <nvtx3/nvToolsExt.h>
#  define NVTX_PUSH(name) nvtxRangePushA(name)
#  define NVTX_POP()      nvtxRangePop()
#else
#  define NVTX_PUSH(name) ((void)0)
#  define NVTX_POP()      ((void)0)
#endif
