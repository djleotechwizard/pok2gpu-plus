#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "../emu/gb.h"
#include "../analysis/cfg.h"
#include "../analysis/ir_interp.h"

/*
 * GBBatch — N independent Game Boy instances sharing a single read-only ROM
 * and a single IR block cache / CFG.  All instances start from the same
 * initial state and evolve independently.
 *
 * Determinism guarantee: given the same ROM and the same initial CPU state,
 * all instances follow identical execution paths.
 * batch_verify_determinism() checks that every instance matches instance 0.
 */

typedef struct {
    GB      *states;
    int      n_states;

    uint8_t *shared_rom;
    uint32_t rom_size;
    uint32_t ext_ram_size;

    CFG     *cfg;
    IRCache  cache;
} GBBatch;

GBBatch *batch_create(int n_states, const char *rom_path);
void     batch_free(GBBatch *b);

void batch_step(GBBatch *b);
void batch_run_frames(GBBatch *b, int n_frames);

int  batch_verify_determinism(const GBBatch *b);
void batch_print_stats(const GBBatch *b);
