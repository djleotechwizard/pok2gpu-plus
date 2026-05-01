/*
 * test_cpu.c — SM83 CPU opcode correctness tests.
 */

#include "harness.h"
#include "../emu/gb.h"
#include <string.h>

static uint8_t s_rom[0x8000];

static GB make_gb(void) {
    GB gb;
    gb_init(&gb);
    memset(s_rom, 0x00, sizeof(s_rom));
    s_rom[0x147] = 0x00;
    s_rom[0x148] = 0x00;
    s_rom[0x149] = 0x00;
    gb.mem.rom      = s_rom;
    gb.mem.rom_size = sizeof(s_rom);
    gb.mem.mbc_type = MBC_NONE;
    cpu_init(&gb.cpu);
    return gb;
}

static void test_nop(void) {
    GB gb = make_gb();
    uint16_t pc0 = gb.cpu.PC;
    s_rom[pc0] = 0x00;
    int cyc = cpu_step(&gb);
    ASSERT_EQ(gb.cpu.PC, pc0 + 1);
    ASSERT_EQ(cyc, 4);
}

static void test_ld_b_imm8(void) {
    GB gb = make_gb();
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0x06;
    s_rom[pc+1] = 0x42;
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.B, 0x42);
}

static void test_xor_a(void) {
    GB gb = make_gb();
    gb.cpu.A = 0x55;
    s_rom[gb.cpu.PC] = 0xAF;
    cpu_step(&gb);
    ASSERT_EQ(gb.cpu.A, 0);
    ASSERT_EQ(GET_Z(&gb.cpu), 1);
    ASSERT_EQ(GET_N(&gb.cpu), 0);
    ASSERT_EQ(GET_H(&gb.cpu), 0);
    ASSERT_EQ(GET_C(&gb.cpu), 0);
}

static void test_add_a_b_no_carry(void) {
    GB gb = make_gb();
    gb.cpu.A = 0x0F;
    gb.cpu.B = 0x01;
    s_rom[gb.cpu.PC] = 0x80;
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.A, 0x10);
    ASSERT_EQ(GET_Z(&gb.cpu), 0);
    ASSERT_EQ(GET_H(&gb.cpu), 1);
    ASSERT_EQ(GET_C(&gb.cpu), 0);
}

static void test_add_a_b_overflow(void) {
    GB gb = make_gb();
    gb.cpu.A = 0xFF;
    gb.cpu.B = 0x01;
    s_rom[gb.cpu.PC] = 0x80;
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.A, 0x00);
    ASSERT_EQ(GET_Z(&gb.cpu), 1);
    ASSERT_EQ(GET_C(&gb.cpu), 1);
}

static void test_inc_b(void) {
    GB gb = make_gb();
    gb.cpu.B = 0x0F;
    s_rom[gb.cpu.PC] = 0x04;
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.B, 0x10);
    ASSERT_EQ(GET_H(&gb.cpu), 1);
    ASSERT_EQ(GET_N(&gb.cpu), 0);
}

static void test_dec_b(void) {
    GB gb = make_gb();
    gb.cpu.B = 0x05;
    s_rom[gb.cpu.PC] = 0x05;
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.B, 0x04);
    ASSERT_EQ(GET_N(&gb.cpu), 1);
}

static void test_jr_nz_taken(void) {
    GB gb = make_gb();
    gb.cpu.F &= ~FLAG_Z;
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0x20;
    s_rom[pc+1] = 0x02;
    int cyc = cpu_step(&gb);
    ASSERT_EQ(gb.cpu.PC, pc + 2 + 2);
    ASSERT_EQ(cyc, 12);
}

static void test_jr_nz_not_taken(void) {
    GB gb = make_gb();
    gb.cpu.F |= FLAG_Z;
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0x20;
    s_rom[pc+1] = 0x02;
    int cyc = cpu_step(&gb);
    ASSERT_EQ(gb.cpu.PC, pc + 2);
    ASSERT_EQ(cyc, 8);
}

static void test_push_pop_roundtrip(void) {
    GB gb = make_gb();
    gb.cpu.B = 0xDE;
    gb.cpu.C = 0xAD;
    gb.cpu.SP = 0xFFFE;
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0xC5; /* PUSH BC */
    s_rom[pc+1] = 0xD1; /* POP DE */
    cpu_step(&gb);
    cpu_step(&gb);
    ASSERT_EQ_HEX(gb.cpu.D, 0xDE);
    ASSERT_EQ_HEX(gb.cpu.E, 0xAD);
    ASSERT_EQ(gb.cpu.SP, 0xFFFE);
}

static void test_cb_bit7_a_zero(void) {
    GB gb = make_gb();
    gb.cpu.A = 0x3F;
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0xCB;
    s_rom[pc+1] = 0x7F;
    cpu_step(&gb);
    ASSERT_EQ(GET_Z(&gb.cpu), 1);
    ASSERT_EQ(GET_H(&gb.cpu), 1);
    ASSERT_EQ(GET_N(&gb.cpu), 0);
}

static void test_ld_hl_imm16(void) {
    GB gb = make_gb();
    uint16_t pc = gb.cpu.PC;
    s_rom[pc]   = 0x21;
    s_rom[pc+1] = 0x34;
    s_rom[pc+2] = 0x12;
    cpu_step(&gb);
    ASSERT_EQ_HEX(cpu_HL(&gb.cpu), 0x1234);
}

int main(void) {
    RUN(test_nop);
    RUN(test_ld_b_imm8);
    RUN(test_xor_a);
    RUN(test_add_a_b_no_carry);
    RUN(test_add_a_b_overflow);
    RUN(test_inc_b);
    RUN(test_dec_b);
    RUN(test_jr_nz_taken);
    RUN(test_jr_nz_not_taken);
    RUN(test_push_pop_roundtrip);
    RUN(test_cb_bit7_a_zero);
    RUN(test_ld_hl_imm16);
    return DONE();
}
