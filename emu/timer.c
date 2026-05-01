#include "gb.h"

/*
 * Timer — DIV/TIMA/TMA/TAC
 *
 * Internal 16-bit counter increments every T-cycle.
 * DIV (0xFF04) = upper byte of the counter.
 * TIMA increments on a falling edge of the TAC-selected bit.
 *
 * TAC bits:
 *   Bit 2: timer enable
 *   Bit 1-0: clock select
 *     00 = bit 9  (4096 Hz)
 *     01 = bit 3  (262144 Hz)
 *     10 = bit 5  (65536 Hz)
 *     11 = bit 7  (16384 Hz)
 */

static const int tac_bit[4] = { 9, 3, 5, 7 };

#define INT_TIMER 0x04

void timer_init(Timer *t) {
    t->div_internal  = 0xABCC;
    t->tima          = 0;
    t->tma           = 0;
    t->tac           = 0;
    t->tima_overflow = false;
    t->overflow_delay = 0;
}

static inline bool tima_clock_bit(const Timer *t) {
    int bit = tac_bit[t->tac & 0x03];
    return (t->div_internal >> bit) & 1;
}

void timer_step(GB *gb, int cycles) {
    Timer *t = &gb->timer;

    if (t->tima_overflow) {
        t->overflow_delay -= cycles;
        if (t->overflow_delay <= 0) {
            t->tima_overflow = false;
            t->tima = t->tma;
            gb->mem.io[0x0F] |= INT_TIMER;
        }
    }

    bool timer_enabled = (t->tac & 0x04) != 0;

    for (int i = 0; i < cycles; i++) {
        bool old_bit = timer_enabled ? tima_clock_bit(t) : false;
        t->div_internal++;
        bool new_bit = timer_enabled ? tima_clock_bit(t) : false;

        if (old_bit && !new_bit) {
            t->tima++;
            if (t->tima == 0) {
                t->tima_overflow  = true;
                t->overflow_delay = 4;
            }
        }
    }

    gb->mem.io[0x04] = (uint8_t)(t->div_internal >> 8);
}
