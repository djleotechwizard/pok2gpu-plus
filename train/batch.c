#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "batch.h"

GBBatch *batch_create(int n_states, const char *rom_path)
{
    if (n_states <= 0) return NULL;

    GBBatch *b = (GBBatch *)calloc(1, sizeof(*b));
    if (!b) return NULL;

    b->n_states = n_states;
    b->states   = (GB *)calloc((size_t)n_states, sizeof(GB));
    if (!b->states) { free(b); return NULL; }

    gb_init(&b->states[0]);
    if (gb_load_rom(&b->states[0], rom_path) != 0) {
        free(b->states); free(b); return NULL;
    }

    b->shared_rom   = b->states[0].mem.rom;
    b->rom_size     = b->states[0].mem.rom_size;
    b->ext_ram_size = b->states[0].mem.ext_ram_size;

    for (int i = 1; i < n_states; i++) {
        b->states[i] = b->states[0];

        if (b->ext_ram_size > 0) {
            b->states[i].mem.ext_ram = (uint8_t *)malloc(b->ext_ram_size);
            if (!b->states[i].mem.ext_ram) {
                for (int j = 1; j < i; j++) free(b->states[j].mem.ext_ram);
                free(b->states[0].mem.rom);
                if (b->states[0].mem.ext_ram) free(b->states[0].mem.ext_ram);
                free(b->states); free(b); return NULL;
            }
            memcpy(b->states[i].mem.ext_ram,
                   b->states[0].mem.ext_ram,
                   b->ext_ram_size);
        }
    }

    fprintf(stderr, "Building CFG for batch (%d states)...\n", n_states);
    b->cfg = cfg_build(b->shared_rom, b->rom_size);
    ir_cache_init(&b->cache);

    return b;
}

void batch_free(GBBatch *b)
{
    if (!b) return;
    for (int i = 1; i < b->n_states; i++)
        if (b->states[i].mem.ext_ram) free(b->states[i].mem.ext_ram);
    if (b->states[0].mem.rom)     free(b->states[0].mem.rom);
    if (b->states[0].mem.ext_ram) free(b->states[0].mem.ext_ram);
    ir_cache_free(&b->cache);
    cfg_free(b->cfg);
    free(b->states);
    free(b);
}

void batch_step(GBBatch *b)
{
    for (int i = 0; i < b->n_states; i++) {
        GB *gb     = &b->states[i];
        int cycles = ir_step(gb, &b->cache, b->cfg);
        ppu_step(gb, cycles);
        timer_step(gb, cycles);
        memory_dma_tick(gb, cycles);
    }
}

void batch_run_frames(GBBatch *b, int n_frames)
{
    uint32_t start  = b->states[0].ppu.frame;
    uint32_t target = start + (uint32_t)n_frames;
    uint32_t last_reported = start;
    uint64_t step_count = 0;

    while (b->states[0].ppu.frame < target) {
        batch_step(b);
        step_count++;

        uint32_t cur = b->states[0].ppu.frame;
        if (cur != last_reported) {
            last_reported = cur;
            fprintf(stderr,
                "\r  frame %u/%u  PC=%04X  steps=%llu  hits=%d misses=%d   ",
                cur - start, (uint32_t)n_frames,
                b->states[0].cpu.PC,
                (unsigned long long)step_count,
                b->cache.hits, b->cache.misses);
            fflush(stderr);
        }
    }
    fprintf(stderr, "\n");
}

int batch_verify_determinism(const GBBatch *b)
{
    if (b->n_states <= 1) return 0;
    const GB *ref = &b->states[0];
    int diverged  = 0;

    for (int i = 1; i < b->n_states; i++) {
        const GB *s = &b->states[i];
        if (s->cpu.PC     != ref->cpu.PC     ||
            s->cpu.cycles != ref->cpu.cycles ||
            s->ppu.frame  != ref->ppu.frame  ||
            s->cpu.A      != ref->cpu.A      ||
            s->cpu.F      != ref->cpu.F      ||
            s->cpu.SP     != ref->cpu.SP) {
            fprintf(stderr,
                "[BATCH DIFF] state %d  PC=%04X (ref=%04X)  "
                "cycles=%llu (ref=%llu)  frame=%u (ref=%u)\n",
                i,
                s->cpu.PC,    ref->cpu.PC,
                (unsigned long long)s->cpu.cycles,
                (unsigned long long)ref->cpu.cycles,
                s->ppu.frame, ref->ppu.frame);
            diverged++;
        }
    }
    return diverged;
}

void batch_print_stats(const GBBatch *b)
{
    const GB *s0 = &b->states[0];
    fprintf(stderr,
        "Batch: %d states  frame=%u  PC=%04X  cycles=%llu  "
        "cache hits=%d misses=%d\n",
        b->n_states, s0->ppu.frame, s0->cpu.PC,
        (unsigned long long)s0->cpu.cycles,
        b->cache.hits, b->cache.misses);
}
