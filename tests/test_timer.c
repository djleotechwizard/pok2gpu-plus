/*
 * test_timer.c — Timer (DIV/TIMA/TMA/TAC) accuracy tests.
 */

#include "harness.h"
#include "../emu/gb.h"
#include <string.h>

static GB make_gb(void) {
    GB gb;
    gb_init(&gb);
    return gb;
}

static void test_div_advances(void) {
    GB gb = make_gb();
    gb.timer.div_internal = 0;
    timer_step(&gb, 256);
    ASSERT_EQ((gb.timer.div_internal >> 8), 1);
}

static void test_tima_4096hz(void) {
    GB gb = make_gb();
    gb.timer.tac = 0x04;
    gb.timer.div_internal = 0;
    gb.timer.tima = 0;
    timer_step(&gb, 1024);
    ASSERT_EQ(gb.timer.tima, 1);
}

static void test_tima_overflow_reload(void) {
    GB gb = make_gb();
    gb.timer.tac  = 0x05; /* 262144 Hz, bit 3, period=16 */
    gb.timer.tima = 0xFF;
    gb.timer.tma  = 0x42;
    gb.timer.div_internal = 0;
    timer_step(&gb, 16);
    ASSERT_EQ(gb.timer.tima_overflow, true);
    timer_step(&gb, 4);
    ASSERT_EQ(gb.timer.tima_overflow, false);
    ASSERT_EQ_HEX(gb.timer.tima, 0x42);
    ASSERT_EQ(gb.mem.io[0x0F] & 0x04, 0x04);
}

static void test_tima_disabled(void) {
    GB gb = make_gb();
    gb.timer.tac = 0x00;
    gb.timer.tima = 0;
    timer_step(&gb, 8192);
    ASSERT_EQ(gb.timer.tima, 0);
}

static void test_div_write_resets(void) {
    GB gb = make_gb();
    gb.timer.tac = 0x04;
    gb.timer.div_internal = (1 << 9) - 1;
    memory_write(&gb, 0xFF04, 0);
    ASSERT_EQ(gb.timer.div_internal, 0);
}

static void test_tima_16384hz(void) {
    GB gb = make_gb();
    gb.timer.tac = 0x07; /* 16384 Hz, bit 7, period=256 */
    gb.timer.div_internal = 0;
    gb.timer.tima = 0;
    timer_step(&gb, 256);
    ASSERT_EQ(gb.timer.tima, 1);
}

int main(void) {
    RUN(test_div_advances);
    RUN(test_tima_4096hz);
    RUN(test_tima_overflow_reload);
    RUN(test_tima_disabled);
    RUN(test_div_write_resets);
    RUN(test_tima_16384hz);
    return DONE();
}
