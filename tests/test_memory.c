/*
 * test_memory.c — Memory map and MBC banking tests.
 */

#include "harness.h"
#include "../emu/gb.h"
#include <string.h>

static uint8_t s_rom[0x8000 * 4];

static GB make_gb_mbc1(void) {
    GB gb;
    gb_init(&gb);
    memset(s_rom, 0xFF, sizeof(s_rom));
    for (int bank = 0; bank < 8; bank++)
        s_rom[(size_t)bank * 0x4000] = (uint8_t)(bank * 0x10);
    s_rom[0x147] = 0x01;
    s_rom[0x148] = 0x03;
    s_rom[0x149] = 0x00;
    gb.mem.rom      = s_rom;
    gb.mem.rom_size = sizeof(s_rom);
    gb.mem.mbc_type = MBC_1;
    gb.mem.rom_bank = 1;
    return gb;
}

static void test_bank0_sentinel(void) {
    GB gb = make_gb_mbc1();
    uint8_t v = memory_read(&gb, 0x0000);
    ASSERT_EQ_HEX(v, 0x00);
}

static void test_mbc1_bank_select(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0x2000, 0x02);
    ASSERT_EQ(gb.mem.rom_bank, 2);
    uint8_t v = memory_read(&gb, 0x4000);
    ASSERT_EQ_HEX(v, 0x20);
}

static void test_mbc1_bank0_promotes_to_1(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0x2000, 0x00);
    ASSERT_EQ(gb.mem.rom_bank, 1);
}

static void test_wram_roundtrip(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0xC100, 0xAB);
    ASSERT_EQ_HEX(memory_read(&gb, 0xC100), 0xAB);
}

static void test_wram_echo(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0xC010, 0x55);
    ASSERT_EQ_HEX(memory_read(&gb, 0xE010), 0x55);
}

static void test_hram_roundtrip(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0xFF90, 0x77);
    ASSERT_EQ_HEX(memory_read(&gb, 0xFF90), 0x77);
}

static void test_ie_roundtrip(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0xFFFF, 0x1F);
    ASSERT_EQ_HEX(memory_read(&gb, 0xFFFF), 0x1F);
}

static void test_if_upper_bits(void) {
    GB gb = make_gb_mbc1();
    memory_write(&gb, 0xFF0F, 0x00);
    ASSERT_EQ_HEX(memory_read(&gb, 0xFF0F) & 0xE0, 0xE0);
}

static void test_div_reset(void) {
    GB gb = make_gb_mbc1();
    gb.timer.div_internal = 0xABCD;
    memory_write(&gb, 0xFF04, 0x42);
    ASSERT_EQ(gb.timer.div_internal, 0);
}

static void test_unusable_region(void) {
    GB gb = make_gb_mbc1();
    ASSERT_EQ_HEX(memory_read(&gb, 0xFEA0), 0xFF);
}

int main(void) {
    RUN(test_bank0_sentinel);
    RUN(test_mbc1_bank_select);
    RUN(test_mbc1_bank0_promotes_to_1);
    RUN(test_wram_roundtrip);
    RUN(test_wram_echo);
    RUN(test_hram_roundtrip);
    RUN(test_ie_roundtrip);
    RUN(test_if_upper_bits);
    RUN(test_div_reset);
    RUN(test_unusable_region);
    return DONE();
}
