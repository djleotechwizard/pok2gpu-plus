#include <string.h>
#include "gb.h"

void gb_init(GB *gb) {
    memset(gb, 0, sizeof(*gb));
    memory_init(&gb->mem);
    cpu_init(&gb->cpu);
    ppu_init(&gb->ppu);
    timer_init(&gb->timer);
    gb->tracer.enabled = false;
    gb->tracer.file    = NULL;
}

int gb_load_rom(GB *gb, const char *path) {
    return memory_load_rom(&gb->mem, path);
}

void gb_enable_trace(GB *gb, const char *path) {
    tracer_init(&gb->tracer, path);
}

void gb_disable_trace(GB *gb) {
    tracer_close(&gb->tracer);
}

void gb_step(GB *gb) {
    int cycles = cpu_step(gb);
    ppu_step(gb, cycles);
    timer_step(gb, cycles);
    memory_dma_tick(gb, cycles);
}

void gb_run_frames(GB *gb, int n) {
    uint32_t target = gb->ppu.frame + (uint32_t)n;
    while (gb->ppu.frame < target)
        gb_step(gb);
}
