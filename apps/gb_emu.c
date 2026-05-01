#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../emu/gb.h"

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <rom.gb>\n"
        "Options:\n"
        "  -f <n>         Run for n frames (default: 600)\n"
        "  -t <file>      Write instruction trace to <file>\n"
        "  -s <n>         Run for n steps (single instructions) instead of frames\n"
        "  -F <n>         Dump a PPM frame every n steps (requires -s)\n"
        "  -v             Print per-frame progress\n"
        "  -h             Show this help\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *rom_path   = NULL;
    const char *trace_path = NULL;
    int   frames       = 600;
    int   steps        = -1;   /* -1 = use frames */
    int   frame_every  = -1;   /* dump PPM every N steps, -1 = off */
    bool  verbose      = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "-v") == 0) { verbose = true; }
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) { frames      = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) { steps       = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-F") == 0 && i+1 < argc) { frame_every = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) { trace_path  = argv[++i]; }
        else if (argv[i][0] != '-') { rom_path = argv[i]; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (!rom_path) {
        fprintf(stderr, "Error: no ROM file specified.\n");
        usage(argv[0]);
        return 1;
    }

    GB gb;
    gb_init(&gb);

    if (gb_load_rom(&gb, rom_path) != 0) return 1;

    if (trace_path) {
        gb_enable_trace(&gb, trace_path);
        fprintf(stderr, "Tracing to: %s\n", trace_path);
    }

    fprintf(stderr, "Starting emulation...\n");

    if (steps > 0) {
        /* Step mode: execute exactly N instructions */
        int dump_idx = 0;
        for (int i = 0; i < steps; i++) {
            gb_step(&gb);
            if (frame_every > 0 && (i + 1) % frame_every == 0) {
                char path[64];
                snprintf(path, sizeof(path), "pics/frame_%06d.ppm", dump_idx++);
                ppu_save_ppm(&gb.ppu, path);
                if (verbose)
                    fprintf(stderr, "Saved %s  (step %d, PPU frame %u)\n",
                            path, i + 1, gb.ppu.frame);
            }
        }
        fprintf(stderr, "Executed %d steps, total cycles: %llu\n",
                steps, (unsigned long long)gb.cpu.cycles);
    } else {
        /* Frame mode */
        uint32_t start_frame = gb.ppu.frame;
        for (int f = 0; f < frames; f++) {
            uint32_t next = gb.ppu.frame + 1;
            while (gb.ppu.frame < next) gb_step(&gb);
            if (verbose && (f % 60 == 0)) {
                fprintf(stderr, "Frame %u  PC=%04X  cycles=%llu\n",
                        gb.ppu.frame,
                        gb.cpu.PC,
                        (unsigned long long)gb.cpu.cycles);
            }
        }
        fprintf(stderr, "Done. Ran %u frames (%llu T-cycles).  PC=%04X\n",
                gb.ppu.frame - start_frame,
                (unsigned long long)gb.cpu.cycles,
                gb.cpu.PC);
    }

    if (trace_path) {
        gb_disable_trace(&gb);
        fprintf(stderr, "Trace entries written: %llu\n",
                (unsigned long long)gb.tracer.entry_count);
    }

    return 0;
}
