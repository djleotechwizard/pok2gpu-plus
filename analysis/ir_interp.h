#pragma once
#include <stdint.h>
#include "ir.h"
#include "../emu/gb.h"

/*
 * IR interpreter — executes IRBlocks against real GB state.
 *
 * Block cache: keyed by (bank<<16)|entry_addr.
 * 65536-slot open-addressing hash table; blocks are lifted once and
 * cached for the lifetime of the process (ROM is static).
 */

#define IR_CACHE_SLOTS  65536

typedef struct {
    uint32_t  key;
    IRBlock  *block;
} IRCacheEntry;

typedef struct {
    IRCacheEntry slots[IR_CACHE_SLOTS];
    int          hits;
    int          misses;
} IRCache;

void           ir_cache_init(IRCache *c);
void           ir_cache_free(IRCache *c);

const IRBlock *ir_cache_get(IRCache *c, const CFG *cfg, int bank, uint16_t addr);

int ir_exec_block(GB *gb, const IRBlock *ib);

/* High-level step: interrupt check → block lookup → execute.
 * Falls back to cpu_step() on cache miss. */
int ir_step(GB *gb, IRCache *cache, const CFG *cfg);
