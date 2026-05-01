#pragma once
#include <stdint.h>
#include <stdio.h>
#include "disasm.h"

/*
 * Basic block and CFG types.
 *
 * A basic block is a maximal straight-line sequence of instructions:
 *   entered only at its first instruction, exited only at its last.
 *
 * Exit types map to the CFType of the terminating instruction.
 */

typedef enum {
    EXIT_FALLTHROUGH,
    EXIT_JUMP,
    EXIT_BRANCH,
    EXIT_CALL,
    EXIT_CALL_CC,
    EXIT_RET,
    EXIT_HALT,
    EXIT_INDIRECT,
} ExitType;

#define BLOCK_MAX_INSNS 512

typedef struct {
    uint16_t entry;
    int      bank;

    Insn     insns[BLOCK_MAX_INSNS];
    int      n_insns;

    ExitType exit_type;
    uint16_t succs[2];
    int      n_succs;
} Block;

typedef struct {
    Block   *blocks;
    int      n_blocks;
    int      cap;
} CFG;

/*
 * cfg_build — recursive-descent CFG construction from ROM image.
 * Returns a heap-allocated CFG that must be freed with cfg_free().
 */
CFG *cfg_build(const uint8_t *rom, uint32_t rom_size);

void cfg_free(CFG *cfg);

void cfg_write_json(const CFG *cfg, FILE *f, const char *rom_path);

void cfg_print_stats(const CFG *cfg);

const char *exit_name(ExitType e);
