#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "batch.h"

/* ============================================================
 * GPU RL exploration
 *
 * Reward: unique (map, y, x) tiles visited per episode.
 * Algorithm: population-based evolutionary strategy.
 * ============================================================ */

#define RL_EPISODE_FRAMES    600
#define RL_ACTION_REPEAT       8
#define RL_N_ACTIONS           9

/* Per-state tile hash set: 512 slots (2 KB/agent).
 * Handles up to ~380 action_steps at <75% load. */
#define RL_TILE_HASH_BITS    9
#define RL_TILE_HASH_SIZE   (1 << RL_TILE_HASH_BITS)
#define RL_TILE_HASH_EMPTY  0xFFFFFFFFu

/* Sentinel inserted when entering a new map (+1 for the map transition) */
#define RL_MAP_SENTINEL     0xFFFFu

/* Action index → joypad bitmask
 * bit0=A  bit1=B  bit2=sel  bit3=start
 * bit4=right  bit5=left  bit6=up  bit7=down  (1 = pressed)
 * 0=nothing 1=up 2=down 3=left 4=right 5=A 6=B 7=start 8=select */
static const uint8_t RL_ACTION_MAP[RL_N_ACTIONS] = {
    0x00, 0x40, 0x80, 0x20, 0x10, 0x01, 0x02, 0x08, 0x04,
};

/* ── RL batch ──────────────────────────────────────────────── */

typedef struct {
    GPUBatch   *batch;
    int         n_states;
    int         ep_frames;
    int         action_steps;
    int         action_repeat;

    uint8_t    *d_act_seqs;          /* [n_states × action_steps] */
    uint32_t   *d_tile_sets;         /* [n_states × RL_TILE_HASH_SIZE] */
    uint32_t   *d_tile_counts;       /* [n_states] */

    /* Episode-reset snapshot (single template; broadcast at reset) */
    GBStateGPU *d_init_states;
    uint8_t    *d_init_ext_ram;
    uint8_t    *d_init_vram;
    uint8_t    *d_init_wram;

    uint8_t    *d_alive;             /* Successive Halving mask; NULL = all alive */

    /* Host mirrors */
    uint32_t   *h_tile_counts;
    uint8_t    *h_act_seqs;
} RLBatch;

/* ── API ───────────────────────────────────────────────────── */

RLBatch *rl_batch_create(GPUBatch *batch, int ep_frames, int action_repeat);
void     rl_batch_free(RLBatch *r);

void rl_batch_set_seqs(RLBatch *r, const uint8_t *h_seqs);

void rl_episode_run(RLBatch *r);
void rl_episode_get_counts(RLBatch *r);
void rl_episode_set_alive(RLBatch *r, const uint8_t *host_alive);

#ifdef __cplusplus
}
#endif
