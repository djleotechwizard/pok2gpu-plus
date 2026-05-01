#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cfg.h"

/* ============================================================
 * CFG construction — two-pass algorithm
 *
 * Pass 1: Discover all basic block entry points via a worklist.
 *         Entry points are seeded by known reset/interrupt vectors
 *         and extended by following control flow.
 *
 * Pass 2: For each entry point, decode instructions until
 *         a block terminator or another entry point is reached,
 *         then record the block and its successors.
 * ============================================================ */

const char *exit_name(ExitType e) {
    switch (e) {
        case EXIT_FALLTHROUGH: return "FALLTHROUGH";
        case EXIT_JUMP:        return "JUMP";
        case EXIT_BRANCH:      return "BRANCH";
        case EXIT_CALL:        return "CALL";
        case EXIT_CALL_CC:     return "CALL_CC";
        case EXIT_RET:         return "RET";
        case EXIT_HALT:        return "HALT";
        case EXIT_INDIRECT:    return "INDIRECT";
        default:               return "?";
    }
}

/* ---- Address-set (bit array over 0x0000–0x7FFF) ---- */

#define ADDRSET_SIZE (0x8000 / 8)

typedef struct { uint8_t bits[ADDRSET_SIZE]; } AddrSet;

static inline void  as_set(AddrSet *s, uint16_t a)        { s->bits[a>>3] |=  (1u << (a&7)); }
static inline void  as_clr(AddrSet *s, uint16_t a)        { s->bits[a>>3] &= ~(1u << (a&7)); }
static inline bool  as_get(const AddrSet *s, uint16_t a)  { return (s->bits[a>>3] >> (a&7)) & 1; }

/* ---- Simple worklist (queue of uint16_t) ---- */

#define WL_CAP_INIT 4096

typedef struct {
    uint16_t *data;
    int       head, tail, cap;
} Worklist;

static Worklist *wl_new(void) {
    Worklist *w = calloc(1, sizeof(*w));
    w->data = malloc(WL_CAP_INIT * sizeof(uint16_t));
    w->cap  = WL_CAP_INIT;
    return w;
}

static void wl_free(Worklist *w) { free(w->data); free(w); }

static bool wl_empty(const Worklist *w) { return w->head == w->tail; }

static void wl_push(Worklist *w, uint16_t a) {
    int next = (w->tail + 1) % w->cap;
    if (next == w->head) {
        /* grow */
        int new_cap = w->cap * 2;
        uint16_t *nd = malloc(new_cap * sizeof(uint16_t));
        int i = 0;
        for (int j = w->head; j != w->tail; j = (j+1) % w->cap) nd[i++] = w->data[j];
        free(w->data);
        w->data = nd;
        w->head = 0;
        w->tail = i;
        w->cap  = new_cap;
        next    = (w->tail + 1) % w->cap;
    }
    w->data[w->tail] = a;
    w->tail = (w->tail + 1) % w->cap;
}

static uint16_t wl_pop(Worklist *w) {
    uint16_t v = w->data[w->head];
    w->head = (w->head + 1) % w->cap;
    return v;
}

/* ---- CFG block array helpers ---- */

static Block *cfg_alloc_block(CFG *cfg) {
    if (cfg->n_blocks >= cfg->cap) {
        cfg->cap = cfg->cap ? cfg->cap * 2 : 256;
        cfg->blocks = realloc(cfg->blocks, (size_t)cfg->cap * sizeof(Block));
    }
    Block *b = &cfg->blocks[cfg->n_blocks++];
    memset(b, 0, sizeof(*b));
    return b;
}

/* ---- Address validation: only follow addresses in ROM range ---- */

static inline bool valid_rom_addr(uint16_t a) {
    return a < 0x8000;
}

static inline bool valid_target(uint16_t t) {
    /* Don't follow targets into I/O or RAM (>= 0x8000) */
    return t < 0x8000;
}

/* ---- Build flat 0x8000 view: bank 0 + selected switchable bank ---- */

static void build_view(const uint8_t *rom, uint32_t rom_size,
                        int bank, uint8_t view[0x8000]) {
    /* Bank 0: ROM bytes 0x0000–0x3FFF (always fixed) */
    uint32_t b0_size = (rom_size >= 0x4000) ? 0x4000 : rom_size;
    memcpy(view, rom, b0_size);
    if (b0_size < 0x4000) memset(view + b0_size, 0xFF, 0x4000 - b0_size);

    /* Switchable bank: occupies view[0x4000..0x7FFF] */
    uint32_t bank_offset = (uint32_t)bank * 0x4000;
    uint32_t bank_avail  = (bank_offset < rom_size)
                           ? (uint32_t)(rom_size - bank_offset) : 0;
    uint32_t copy_len    = (bank_avail >= 0x4000) ? 0x4000 : bank_avail;
    if (copy_len > 0) memcpy(view + 0x4000, rom + bank_offset, copy_len);
    if (copy_len < 0x4000) memset(view + 0x4000 + copy_len, 0xFF, 0x4000 - copy_len);
}

/* ---- Pass 1: collect all block entry points in this bank view ---- */

static void collect_entries(const uint8_t *view, AddrSet *entries,
                             const uint16_t *seeds, int n_seeds) {
    AddrSet visited;
    memset(&visited, 0, sizeof(visited));

    Worklist *wl = wl_new();

    /* Seed initial entry points */
    for (int i = 0; i < n_seeds; i++) {
        uint16_t a = seeds[i];
        if (!as_get(entries, a)) {
            as_set(entries, a);
            wl_push(wl, a);
        }
    }

    while (!wl_empty(wl)) {
        uint16_t addr = wl_pop(wl);
        if (as_get(&visited, addr)) continue;
        as_set(&visited, addr);

        uint16_t cur = addr;

        while (valid_rom_addr(cur)) {
            Insn insn;
            disasm_decode(view, cur, &insn);
            uint16_t next = (uint16_t)(cur + insn.len);

            switch (insn.cf) {
                case CF_HALT:
                case CF_RET:
                    goto block_done; /* no successors */

                case CF_INDIRECT:
                    goto block_done; /* can't follow */

                case CF_JUMP:
                    /* Single successor = target */
                    if (valid_target(insn.target) && !as_get(entries, insn.target)) {
                        as_set(entries, insn.target);
                        wl_push(wl, insn.target);
                    }
                    goto block_done;

                case CF_RET_CC:
                    /* Fallthrough is a new entry; don't add the return destination */
                    if (valid_rom_addr(next) && !as_get(entries, next)) {
                        as_set(entries, next);
                        wl_push(wl, next);
                    }
                    goto block_done;

                case CF_BRANCH:
                    /* Two successors: taken target + fallthrough */
                    if (valid_target(insn.target) && !as_get(entries, insn.target)) {
                        as_set(entries, insn.target);
                        wl_push(wl, insn.target);
                    }
                    if (valid_rom_addr(next) && !as_get(entries, next)) {
                        as_set(entries, next);
                        wl_push(wl, next);
                    }
                    goto block_done;

                case CF_CALL:
                    /* Callee is a new entry; fallthrough (return addr) is also a new entry */
                    if (valid_target(insn.target) && !as_get(entries, insn.target)) {
                        as_set(entries, insn.target);
                        wl_push(wl, insn.target);
                    }
                    /* Fallthrough = next instruction (return address) */
                    if (valid_rom_addr(next) && !as_get(entries, next)) {
                        as_set(entries, next);
                        wl_push(wl, next);
                    }
                    goto block_done;

                case CF_CALL_CC:
                    /* Conditional call: same as BRANCH + callee entry */
                    if (valid_target(insn.target) && !as_get(entries, insn.target)) {
                        as_set(entries, insn.target);
                        wl_push(wl, insn.target);
                    }
                    if (valid_rom_addr(next) && !as_get(entries, next)) {
                        as_set(entries, next);
                        wl_push(wl, next);
                    }
                    goto block_done;

                case CF_NEXT:
                default:
                    /* If next addr is already a known entry, end this block */
                    if (valid_rom_addr(next) && as_get(entries, next))
                        goto block_done;
                    cur = next;
                    continue;
            }
        }
        block_done:;
    }

    wl_free(wl);
}

/* ---- Pass 2: decode each entry point into a Block ---- */

static void decode_blocks(const uint8_t *view, const AddrSet *entries,
                           int bank, CFG *cfg) {
    for (uint16_t addr = 0; ; addr++) {
        if (!as_get(entries, addr)) {
            if (addr == 0x7FFF) break;
            continue;
        }

        Block *b = cfg_alloc_block(cfg);
        b->entry = addr;
        b->bank  = bank;

        uint16_t cur = addr;

        while (valid_rom_addr(cur) && b->n_insns < BLOCK_MAX_INSNS) {
            Insn *insn = &b->insns[b->n_insns++];
            disasm_decode(view, cur, insn);
            uint16_t next = (uint16_t)(cur + insn->len);

            switch (insn->cf) {
                case CF_HALT:
                    b->exit_type = EXIT_HALT;
                    b->n_succs   = 0;
                    goto done;

                case CF_RET:
                    b->exit_type = EXIT_RET;
                    b->n_succs   = 0;
                    goto done;

                case CF_INDIRECT:
                    b->exit_type = EXIT_INDIRECT;
                    b->n_succs   = 0;
                    goto done;

                case CF_JUMP:
                    b->exit_type = EXIT_JUMP;
                    if (valid_target(insn->target)) {
                        b->succs[0] = insn->target;
                        b->n_succs  = 1;
                    }
                    goto done;

                case CF_RET_CC:
                    b->exit_type = EXIT_BRANCH;
                    /* succs[0]=fallthrough; no static "taken" address for return */
                    b->succs[0]  = next;
                    b->n_succs   = 1;
                    goto done;

                case CF_BRANCH:
                    b->exit_type = EXIT_BRANCH;
                    b->succs[0]  = insn->target;
                    b->succs[1]  = next;
                    b->n_succs   = 2;
                    goto done;

                case CF_CALL:
                    b->exit_type = EXIT_CALL;
                    b->succs[0]  = insn->target; /* callee */
                    b->succs[1]  = next;         /* return address */
                    b->n_succs   = 2;
                    goto done;

                case CF_CALL_CC:
                    b->exit_type = EXIT_CALL_CC;
                    b->succs[0]  = insn->target;
                    b->succs[1]  = next;
                    b->n_succs   = 2;
                    goto done;

                case CF_NEXT:
                default:
                    /* Check if next is another block's entry */
                    if (valid_rom_addr(next) && as_get(entries, next)) {
                        b->exit_type = EXIT_FALLTHROUGH;
                        b->succs[0]  = next;
                        b->n_succs   = 1;
                        goto done;
                    }
                    cur = next;
                    continue;
            }
        }
        /* Fell off end of ROM or hit BLOCK_MAX_INSNS */
        b->exit_type = EXIT_FALLTHROUGH;
        b->n_succs   = 0;
        done:;

        if (addr == 0x7FFF) break;
    }
}

/* ------------------------------------------------------------------ */

CFG *cfg_build(const uint8_t *rom, uint32_t rom_size) {
    CFG *cfg = calloc(1, sizeof(*cfg));

    /* Number of ROM banks */
    int n_banks = (int)(rom_size / 0x4000);
    if (n_banks < 1) n_banks = 1;

    /* Known entry points (fixed across all banks) */
    static const uint16_t seeds[] = {
        0x0100,           /* reset vector */
        0x0040,           /* VBLANK handler */
        0x0048,           /* STAT handler */
        0x0050,           /* Timer handler */
        0x0058,           /* Serial handler */
        0x0060,           /* Joypad handler */
        0x0000,           /* RST 00 */
        0x0008,           /* RST 08 */
        0x0010,           /* RST 10 */
        0x0018,           /* RST 18 */
        0x0020,           /* RST 20 */
        0x0028,           /* RST 28 */
        0x0030,           /* RST 30 */
        0x0038,           /* RST 38 */
    };
    int n_seeds = (int)(sizeof(seeds) / sizeof(seeds[0]));

    uint8_t *view = malloc(0x8000);

    /* Analyse bank 0 paired with each switchable bank */
    for (int bank = 1; bank < n_banks; bank++) {
        build_view(rom, rom_size, bank, view);

        AddrSet entries;
        memset(&entries, 0, sizeof(entries));

        collect_entries(view, &entries, seeds, n_seeds);
        decode_blocks(view, &entries, bank, cfg);
    }

    free(view);
    return cfg;
}

void cfg_free(CFG *cfg) {
    if (cfg) { free(cfg->blocks); free(cfg); }
}

/* ------------------------------------------------------------------ */

void cfg_print_stats(const CFG *cfg) {
    int by_exit[8] = {0};
    int total_insns = 0;
    for (int i = 0; i < cfg->n_blocks; i++) {
        const Block *b = &cfg->blocks[i];
        total_insns += b->n_insns;
        int e = (int)b->exit_type;
        if (e >= 0 && e < 8) by_exit[e]++;
    }
    fprintf(stderr, "CFG stats:\n");
    fprintf(stderr, "  blocks       : %d\n", cfg->n_blocks);
    fprintf(stderr, "  instructions : %d\n", total_insns);
    for (int i = 0; i < 8; i++) {
        if (by_exit[i])
            fprintf(stderr, "  exit %-12s: %d\n", exit_name((ExitType)i), by_exit[i]);
    }
}

/* ------------------------------------------------------------------ */
/* JSON output                                                         */
/* ------------------------------------------------------------------ */

/* Escape a string for JSON (only ASCII mnemonics expected) */
static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        fputc(*s, f);
    }
    fputc('"', f);
}

void cfg_write_json(const CFG *cfg, FILE *f, const char *rom_path) {
    fprintf(f, "{\n");
    fprintf(f, "  \"rom\": ");
    json_str(f, rom_path);
    fprintf(f, ",\n");
    fprintf(f, "  \"block_count\": %d,\n", cfg->n_blocks);
    fprintf(f, "  \"blocks\": [\n");

    for (int i = 0; i < cfg->n_blocks; i++) {
        const Block *b = &cfg->blocks[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"addr\": %u,\n",     b->entry);
        fprintf(f, "      \"addr_hex\": \"0x%04X\",\n", b->entry);
        fprintf(f, "      \"bank\": %d,\n",     b->bank);
        fprintf(f, "      \"exit_type\": \"%s\",\n", exit_name(b->exit_type));

        /* Successors */
        fprintf(f, "      \"successors\": [");
        for (int s = 0; s < b->n_succs; s++) {
            if (s) fprintf(f, ", ");
            fprintf(f, "%u", b->succs[s]);
        }
        fprintf(f, "],\n");

        /* Instructions */
        fprintf(f, "      \"instructions\": [\n");
        for (int k = 0; k < b->n_insns; k++) {
            const Insn *in = &b->insns[k];
            fprintf(f, "        {");
            fprintf(f, "\"addr\": %u, \"addr_hex\": \"0x%04X\", ", in->addr, in->addr);

            /* bytes as hex string */
            fprintf(f, "\"bytes\": \"");
            for (int bi = 0; bi < in->len; bi++) {
                if (bi) fputc(' ', f);
                fprintf(f, "%02X", in->bytes[bi]);
            }
            fprintf(f, "\", ");

            fprintf(f, "\"mnemonic\": ");
            json_str(f, in->mnemonic);
            fprintf(f, ", ");

            fprintf(f, "\"cf\": \"%s\", ", cf_name(in->cf));
            fprintf(f, "\"cycles\": %d", in->cycles);

            if (in->cf != CF_NEXT && in->cf != CF_RET &&
                in->cf != CF_HALT && in->cf != CF_INDIRECT) {
                fprintf(f, ", \"target\": %u", in->target);
            }

            fprintf(f, "}%s\n", (k < b->n_insns - 1) ? "," : "");
        }
        fprintf(f, "      ]\n");
        fprintf(f, "    }%s\n", (i < cfg->n_blocks - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
}
