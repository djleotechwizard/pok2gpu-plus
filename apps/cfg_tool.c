#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../analysis/cfg.h"

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <rom.gb>\n"
        "Options:\n"
        "  -o <file>   Write JSON output to <file>  (default: cfg.json)\n"
        "  -h          Show this help\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *rom_path  = NULL;
    const char *out_path  = "cfg.json";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) { out_path = argv[++i]; }
        else if (argv[i][0] != '-') { rom_path = argv[i]; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (!rom_path) { usage(argv[0]); return 1; }

    /* Load ROM */
    FILE *f = fopen(rom_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open ROM: %s\n", rom_path); return 1; }
    fseek(f, 0, SEEK_END);
    long rom_size = ftell(f);
    rewind(f);
    uint8_t *rom = malloc((size_t)rom_size);
    if (!rom || fread(rom, 1, (size_t)rom_size, f) != (size_t)rom_size) {
        fprintf(stderr, "Failed to read ROM\n"); return 1;
    }
    fclose(f);

    fprintf(stderr, "ROM: %s  (%ld bytes, %ld banks)\n",
            rom_path, rom_size, rom_size / 0x4000);

    /* Build CFG */
    fprintf(stderr, "Building CFG...\n");
    CFG *cfg = cfg_build(rom, (uint32_t)rom_size);
    free(rom);

    cfg_print_stats(cfg);

    /* Write JSON */
    FILE *out = strcmp(out_path, "-") == 0 ? stdout : fopen(out_path, "w");
    if (!out) { fprintf(stderr, "Cannot write output: %s\n", out_path); return 1; }

    cfg_write_json(cfg, out, rom_path);

    if (out != stdout) {
        fclose(out);
        fprintf(stderr, "JSON written to: %s\n", out_path);
    }

    cfg_free(cfg);
    return 0;
}
