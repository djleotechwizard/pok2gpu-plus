#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../analysis/cfg.h"
#include "../analysis/ir.h"

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <rom.gb>\n"
        "Options:\n"
        "  -o <file>   IR output file     (default: ir.txt)\n"
        "  -b <n>      Only lift bank n   (default: 1)\n"
        "  -a <hex>    Only lift block at address (hex, e.g. 0x0100)\n"
        "  -n <count>  Lift at most <count> blocks\n"
        "  -h          Show this help\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *rom_path = NULL;
    const char *out_path = "ir.txt";
    int         bank     = 1;
    int         limit    = 0;       /* 0 = no limit */
    int         filter_addr = -1;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i],"-h")==0) { usage(argv[0]); return 0; }
        else if (strcmp(argv[i],"-o")==0 && i+1<argc) { out_path   = argv[++i]; }
        else if (strcmp(argv[i],"-b")==0 && i+1<argc) { bank       = atoi(argv[++i]); }
        else if (strcmp(argv[i],"-n")==0 && i+1<argc) { limit      = atoi(argv[++i]); }
        else if (strcmp(argv[i],"-a")==0 && i+1<argc) { filter_addr = (int)strtol(argv[++i],NULL,16); }
        else if (argv[i][0]!='-')                      { rom_path   = argv[i]; }
        else { fprintf(stderr,"Unknown option: %s\n",argv[i]); return 1; }
    }
    if (!rom_path) { usage(argv[0]); return 1; }

    /* Load ROM */
    FILE *rf = fopen(rom_path,"rb");
    if (!rf) { fprintf(stderr,"Cannot open ROM: %s\n",rom_path); return 1; }
    fseek(rf,0,SEEK_END); long rsz=ftell(rf); rewind(rf);
    uint8_t *rom = malloc((size_t)rsz);
    if (!rom || fread(rom,1,(size_t)rsz,rf)!=(size_t)rsz) {
        fprintf(stderr,"Failed to read ROM\n"); return 1; }
    fclose(rf);

    fprintf(stderr,"ROM: %s  (%ld bytes)\n",rom_path,rsz);

    /* Build CFG */
    fprintf(stderr,"Building CFG...\n");
    CFG *cfg = cfg_build(rom,(uint32_t)rsz);
    free(rom);
    cfg_print_stats(cfg);

    /* Open output */
    FILE *out = strcmp(out_path,"-")==0 ? stdout : fopen(out_path,"w");
    if (!out) { fprintf(stderr,"Cannot write: %s\n",out_path); return 1; }

    fprintf(out, "; IR — %s  (bank=%d)\n\n", rom_path, bank);

    /* Lift and print */
    int lifted = 0;
    for (int i = 0; i < cfg->n_blocks; i++) {
        const Block *b = &cfg->blocks[i];
        if (b->bank != bank) continue;
        if (filter_addr >= 0 && b->entry != (uint16_t)filter_addr) continue;
        if (limit > 0 && lifted >= limit) break;

        IRBlock *ib = ir_lift_block(b);
        ir_print_block(ib, b, out);
        ir_free_block(ib);
        lifted++;
    }

    if (out != stdout) fclose(out);
    fprintf(stderr,"Lifted %d blocks → %s\n", lifted, out_path);

    cfg_free(cfg);
    return 0;
}
