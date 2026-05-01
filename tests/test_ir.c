/*
 * test_ir.c — IR lift + interpret correctness tests.
 */

#include "harness.h"
#include "../emu/gb.h"
#include "../analysis/cfg.h"
#include "../analysis/ir.h"
#include "../analysis/ir_interp.h"
#include <string.h>
#include <stdlib.h>

static uint8_t s_rom[0x8000];

static GB make_gb(void) {
    GB gb;
    gb_init(&gb);
    memset(s_rom, 0x00, sizeof(s_rom));
    s_rom[0x147] = 0x00;
    gb.mem.rom      = s_rom;
    gb.mem.rom_size = sizeof(s_rom);
    gb.mem.mbc_type = MBC_NONE;
    cpu_init(&gb.cpu);
    return gb;
}

static Block make_block_from_bytes(const uint8_t *bytes, int len, ExitType exit) {
    Block b;
    memset(&b, 0, sizeof(b));
    b.entry     = 0x0100;
    b.bank      = 1;
    b.exit_type = exit;
    b.n_succs   = (exit == EXIT_FALLTHROUGH) ? 1 : 0;
    if (exit == EXIT_FALLTHROUGH) b.succs[0] = (uint16_t)(0x0100 + len);

    Insn insn;
    memset(&insn, 0, sizeof(insn));
    insn.addr   = 0x0100;
    insn.len    = len;
    memcpy(insn.bytes, bytes, (size_t)len);
    insn.cycles = (len == 1) ? 4 : 8;
    insn.cf     = (exit == EXIT_FALLTHROUGH) ? CF_NEXT : CF_RET;
    b.insns[0]  = insn;
    b.n_insns   = 1;

    memcpy(s_rom + 0x0100, bytes, (size_t)len);
    return b;
}

static void test_ir_ld_b_imm8(void) {
    GB gb_ir  = make_gb();
    GB gb_ref = make_gb();

    uint8_t bytes[] = { 0x06, 0x42 };
    Block b = make_block_from_bytes(bytes, 2, EXIT_FALLTHROUGH);
    IRBlock *ib = ir_lift_block(&b);
    ASSERT(ib != NULL);

    ir_exec_block(&gb_ir, ib);
    cpu_step(&gb_ref);

    ASSERT_EQ_HEX(gb_ir.cpu.B, gb_ref.cpu.B);
    ir_free_block(ib);
}

static void test_ir_xor_a(void) {
    GB gb_ir  = make_gb();
    GB gb_ref = make_gb();
    gb_ir.cpu.A  = 0xAB;
    gb_ref.cpu.A = 0xAB;

    uint8_t bytes[] = { 0xAF };
    Block b = make_block_from_bytes(bytes, 1, EXIT_FALLTHROUGH);
    IRBlock *ib = ir_lift_block(&b);
    ASSERT(ib != NULL);

    ir_exec_block(&gb_ir, ib);
    cpu_step(&gb_ref);

    ASSERT_EQ_HEX(gb_ir.cpu.A, gb_ref.cpu.A);
    ASSERT_EQ_HEX(gb_ir.cpu.F, gb_ref.cpu.F);
    ir_free_block(ib);
}

static void test_ir_add_a_b_flags(void) {
    GB gb_ir  = make_gb();
    GB gb_ref = make_gb();
    gb_ir.cpu.A  = gb_ref.cpu.A = 0x0F;
    gb_ir.cpu.B  = gb_ref.cpu.B = 0x01;

    uint8_t bytes[] = { 0x80 };
    Block b = make_block_from_bytes(bytes, 1, EXIT_FALLTHROUGH);
    IRBlock *ib = ir_lift_block(&b);
    ASSERT(ib != NULL);

    ir_exec_block(&gb_ir, ib);
    cpu_step(&gb_ref);

    ASSERT_EQ_HEX(gb_ir.cpu.A, gb_ref.cpu.A);
    ASSERT_EQ_HEX(gb_ir.cpu.F, gb_ref.cpu.F);
    ir_free_block(ib);
}

static void test_ir_inc_hl(void) {
    GB gb_ir  = make_gb();
    GB gb_ref = make_gb();
    cpu_set_HL(&gb_ir.cpu,  0x00FF);
    cpu_set_HL(&gb_ref.cpu, 0x00FF);

    uint8_t bytes[] = { 0x23 };
    Block b = make_block_from_bytes(bytes, 1, EXIT_FALLTHROUGH);
    IRBlock *ib = ir_lift_block(&b);
    ASSERT(ib != NULL);

    ir_exec_block(&gb_ir, ib);
    cpu_step(&gb_ref);

    ASSERT_EQ_HEX(cpu_HL(&gb_ir.cpu), cpu_HL(&gb_ref.cpu));
    ir_free_block(ib);
}

static void test_ir_cache_hit(void) {
    make_gb(); /* initialise s_rom */

    CFG *cfg = (CFG *)calloc(1, sizeof(CFG));
    cfg->blocks  = (Block *)calloc(1, sizeof(Block));
    cfg->n_blocks = 1;
    cfg->cap      = 1;

    uint8_t bytes[] = { 0x00 };
    cfg->blocks[0] = make_block_from_bytes(bytes, 1, EXIT_FALLTHROUGH);
    cfg->blocks[0].bank = 1;

    IRCache cache;
    ir_cache_init(&cache);

    const IRBlock *b1 = ir_cache_get(&cache, cfg, 1, 0x0100);
    const IRBlock *b2 = ir_cache_get(&cache, cfg, 1, 0x0100);

    ASSERT(b1 != NULL);
    ASSERT_EQ((uintptr_t)b1, (uintptr_t)b2);
    ASSERT_EQ(cache.hits,   1);
    ASSERT_EQ(cache.misses, 1);

    ir_cache_free(&cache);
    cfg_free(cfg);
}

int main(void) {
    RUN(test_ir_ld_b_imm8);
    RUN(test_ir_xor_a);
    RUN(test_ir_add_a_b_flags);
    RUN(test_ir_inc_hl);
    RUN(test_ir_cache_hit);
    return DONE();
}
