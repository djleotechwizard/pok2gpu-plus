#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "../analysis/ir.h"
#include "../analysis/cfg.h"
#include "../emu/gb.h"

/* ========================================================
 * Flat, pointer-free GB state for CUDA threads.
 * Each CUDA thread owns one GBStateGPU (~700 bytes).
 * VRAM and WRAM are NOT inline — stored in separate device
 * arrays so the hot state fits in L1 cache per SM.
 * ======================================================== */

#define GPU_WRAM_SIZE  0x2000
#define GPU_HRAM_SIZE  0x007F
#define GPU_VRAM_SIZE  0x2000
#define GPU_OAM_SIZE   0x00A0
#define GPU_IO_SIZE    0x0080

typedef struct {
    /* CPU */
    uint8_t  A, F, B, C, D, E, H, L;
    uint16_t SP, PC;
    uint8_t  IME;
    uint8_t  ime_pending;
    uint8_t  halted;
    uint8_t  halt_bug;
    uint64_t cycles;

    /* MBC */
    uint8_t  mbc_type;    /* 0=none,1,3,5 */
    uint8_t  rom_bank;
    uint8_t  ram_bank;
    uint8_t  ram_enabled;
    int32_t  mbc1_mode;

    uint8_t  IE;

    /* OAM DMA */
    uint8_t  dma_active;
    uint16_t dma_src;
    int32_t  dma_index;   /* GPU: T-cycles remaining; CPU: byte index */

    /* PPU */
    int32_t  ppu_scanline;
    int32_t  ppu_dot;
    int32_t  ppu_mode;
    uint32_t ppu_frame;

    /* Timer */
    uint16_t timer_div;
    uint8_t  timer_tima;
    uint8_t  timer_tma;
    uint8_t  timer_tac;
    uint8_t  timer_overflow;
    int32_t  timer_overflow_delay;

    /* IO registers [0xFF00–0xFF7F] */
    uint8_t  io[GPU_IO_SIZE];

    /* Debug counters */
    uint64_t fallback_cycles;
    uint32_t fallback_count;

    /* Inline memory */
    uint8_t  hram[GPU_HRAM_SIZE];
    uint8_t  joypad_action;
    uint8_t  oam[GPU_OAM_SIZE];
} GBStateGPU;

/* ── O(1) hash table for block lookup ──────────────────── */

#define GPU_BLOCK_HASH_BITS  17
#define GPU_BLOCK_HASH_SIZE  (1 << GPU_BLOCK_HASH_BITS)
#define GPU_BLOCK_HASH_EMPTY 0xFFFFFFFFu
#define GPU_CHAIN_DEPTH 8

typedef struct {
    uint32_t key;
    int32_t  idx;
} BlockHashSlot;

/* ── Serialised IR block ────────────────────────────────── */

typedef struct {
    uint32_t key;
    int32_t  insn_offset;
    int32_t  n_insns;
    int32_t  n_tmps;
    int32_t  taken_cycles;
    int32_t  fall_cycles;
    int32_t  exit_type;
    uint16_t succs[2];
} IRBlockInfoGPU;

/* ── Host-side GPU batch ────────────────────────────────── */

typedef struct {
    /* Device pointers */
    GBStateGPU    *d_states;
    uint8_t       *d_vrams;
    uint8_t       *d_wrams;
    uint8_t       *d_ext_rams;
    const uint8_t *d_rom;
    IRBlockInfoGPU*d_blocks;
    IRInsn        *d_insns;
    BlockHashSlot *d_hash;
    uint8_t       *d_actions;

    /* Sorted-dispatch buffers */
    int           *d_sorted;
    int           *d_sorted_buf;
    uint32_t      *d_keys;
    uint32_t      *d_keys_buf;
    void          *d_cub_tmp;
    size_t         cub_tmp_size;

    /* Dimensions */
    int      n_states;
    uint32_t rom_size;
    uint32_t ext_ram_size;
    int      n_blocks;
    int      n_insns;

    /* Block-hit profiling (NULL when not profiling) */
    uint32_t *d_profile;
    uint32_t *h_profile;
} GPUBatch;

/* ── API ────────────────────────────────────────────────── */

GPUBatch *gpu_batch_create(const CFG *cfg,
                           const GB  *host_states,
                           const uint8_t *host_ext_rams,
                           int    n_states,
                           uint32_t rom_size,
                           uint32_t ext_ram_size);

void gpu_batch_set_actions(GPUBatch *b, const uint8_t *host_actions);
void gpu_batch_get_actions(const GPUBatch *b, uint8_t *host_actions);
void gpu_batch_run_frames(GPUBatch *b, int n_frames);
void gpu_batch_sync_to_host(const GPUBatch *b, GB *host_states, uint8_t *host_ext_rams);
void gpu_batch_print_stats(const GPUBatch *b);
void gpu_batch_free(GPUBatch *b);

/* Block-hit profiling */
void gpu_batch_start_profiling(GPUBatch *b);
void gpu_batch_stop_profiling(GPUBatch *b);
void gpu_batch_dump_profile(const GPUBatch *b, const char *path);

/* State conversion helpers */
void gb_to_gpu_state(const GB *gb, GBStateGPU *g);
void gpu_state_to_gb(const GBStateGPU *g, GB *gb);

#ifdef __cplusplus
}
#endif
