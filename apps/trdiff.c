/*
 * trdiff — trace comparison tool
 *
 * Reads two instruction trace files produced by gb_emu -t and finds the
 * first divergence between them.  Reports the line number, the differing
 * fields, and N lines of surrounding context.
 *
 * Trace format expected:
 *   INT:XX                          (optional, before the T: line it belongs to)
 *   T:000000000000 PC:0100 OP:XX …  A:XX F:XX B:XX C:XX D:XX E:XX H:XX L:XX SP:XXXX  Z:X N:X H:X C:X
 *     W:XXXX=XX                     (optional memory-write lines; 2-space indent)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ─── Parsed instruction record ─────────────────────────────── */

#define MAX_WRITES 16

typedef struct {
    uint64_t cycle;
    uint16_t pc;
    uint8_t  A, F, B, C, D, E, H, L;
    uint16_t SP;
    int      fZ, fN, fH, fC;
    /* memory writes that follow this instruction */
    struct { uint16_t addr; uint8_t val; } writes[MAX_WRITES];
    int      n_writes;
    /* interrupt that fired before this instruction */
    bool     interrupt_fired;
    uint8_t  int_vec;
    /* raw line (for display) */
    char     raw[256];
} TRec;

/* ─── Line reader ────────────────────────────────────────────── */

typedef struct {
    FILE    *f;
    char     buf[512];
    bool     eof;
    uint64_t lineno;
    /* one-line lookahead */
    bool     have_lookahead;
    char     la[512];
} TReader;

static bool tr_getline(TReader *r, char *out, size_t sz) {
    if (r->have_lookahead) {
        r->have_lookahead = false;
        strncpy(out, r->la, sz - 1);
        out[sz - 1] = '\0';
        r->lineno++;
        return true;
    }
    if (r->eof) return false;
    if (!fgets(out, (int)sz, r->f)) { r->eof = true; return false; }
    r->lineno++;
    /* strip trailing newline */
    size_t n = strlen(out);
    while (n > 0 && (out[n-1] == '\n' || out[n-1] == '\r')) out[--n] = '\0';
    return true;
}

static void tr_unget(TReader *r, const char *line) {
    r->have_lookahead = true;
    strncpy(r->la, line, sizeof(r->la) - 1);
    r->la[sizeof(r->la)-1] = '\0';
    r->lineno--;
}

/* Parse one T: record (+ any following W: lines + preceding INT: line).
 * Returns false on EOF.  Skips blank / comment lines. */
static bool tr_next(TReader *r, TRec *out) {
    char line[512];
    memset(out, 0, sizeof(*out));

    for (;;) {
        if (!tr_getline(r, line, sizeof(line))) return false;

        if (line[0] == '\0') continue; /* blank */

        if (strncmp(line, "INT:", 4) == 0) {
            out->interrupt_fired = true;
            out->int_vec = (uint8_t)strtoul(line + 4, NULL, 16);
            continue; /* next line should be T: */
        }

        if (line[0] == 'T' && line[1] == ':') {
            memcpy(out->raw, line, sizeof(out->raw) - 1);
            out->raw[sizeof(out->raw) - 1] = '\0';
            /* Parse:  T:CCCCCCCCCCCC PC:PPPP OP:...  A:XX F:XX B:XX C:XX D:XX E:XX H:XX L:XX SP:XXXX  Z:X N:X H:X C:X */
            unsigned long long cyc;
            unsigned pc, A, F, B, C, D, E, H, L, SP, Z, N, Hf, Cf;
            /* skip the opcode bytes field (variable width) by scanning from known offsets */
            int n = sscanf(line,
                "T:%llu PC:%x OP:%*s %*s %*s A:%x F:%x B:%x C:%x D:%x E:%x H:%x L:%x SP:%x Z:%d N:%d H:%d C:%d",
                &cyc, &pc, &A, &F, &B, &C, &D, &E, &H, &L, &SP, &Z, &N, &Hf, &Cf);
            if (n < 15) {
                /* OP field may be "XX" or "XX XX" or "XX XX XX"; try parsing without consuming OP */
                /* Fallback: find "A:" directly */
                char *ap = strstr(line, " A:");
                if (ap) {
                    sscanf(line, "T:%llu PC:%x", &cyc, &pc);
                    sscanf(ap,
                        " A:%x F:%x B:%x C:%x D:%x E:%x H:%x L:%x SP:%x Z:%d N:%d H:%d C:%d",
                        &A, &F, &B, &C, &D, &E, &H, &L, &SP, &Z, &N, &Hf, &Cf);
                }
            }
            out->cycle = (uint64_t)cyc;
            out->pc    = (uint16_t)pc;
            out->A     = (uint8_t)A;  out->F = (uint8_t)F;
            out->B     = (uint8_t)B;  out->C = (uint8_t)C;
            out->D     = (uint8_t)D;  out->E = (uint8_t)E;
            out->H     = (uint8_t)H;  out->L = (uint8_t)L;
            out->SP    = (uint16_t)SP;
            out->fZ = Z; out->fN = N; out->fH = Hf; out->fC = Cf;

            /* Read following W: lines */
            while (tr_getline(r, line, sizeof(line))) {
                if (strncmp(line, "  W:", 4) == 0) {
                    if (out->n_writes < MAX_WRITES) {
                        unsigned addr, val;
                        sscanf(line + 2, "W:%x=%x", &addr, &val);
                        out->writes[out->n_writes].addr = (uint16_t)addr;
                        out->writes[out->n_writes].val  = (uint8_t)val;
                        out->n_writes++;
                    }
                } else {
                    tr_unget(r, line);
                    break;
                }
            }
            return true;
        }
        /* unknown line type: skip */
    }
}

/* ─── Context ring buffer ────────────────────────────────────── */

#define CTX 5   /* lines of context before divergence */

typedef struct { TRec recs[CTX]; int head; int count; } Ring;

static void ring_push(Ring *r, const TRec *rec) {
    r->recs[r->head] = *rec;
    r->head = (r->head + 1) % CTX;
    if (r->count < CTX) r->count++;
}

static void ring_print(const Ring *r, const char *label) {
    int start = (r->head - r->count + CTX) % CTX;
    for (int i = 0; i < r->count; i++) {
        const TRec *rec = &r->recs[(start + i) % CTX];
        fprintf(stderr, "  %s  T:%012llu PC:%04X  A:%02X F:%02X B:%02X C:%02X D:%02X E:%02X H:%02X L:%02X SP:%04X\n",
                label,
                (unsigned long long)rec->cycle, rec->pc,
                rec->A, rec->F, rec->B, rec->C,
                rec->D, rec->E, rec->H, rec->L, rec->SP);
    }
}

/* ─── Field-level diff ───────────────────────────────────────── */

static void print_diff(uint64_t lineno, const TRec *a, const TRec *b,
                       const char *fa, const char *fb) {
    fprintf(stderr, "\n[DIVERGE] instruction #%llu\n", (unsigned long long)lineno);

#define CHK(field, fmt) \
    if (a->field != b->field) \
        fprintf(stderr, "  DIFF  " #field ":  %s=" fmt "  %s=" fmt "\n", \
                fa, (a->field), fb, (b->field))

    CHK(cycle, "%lu");
    CHK(pc,    "%04X");
    CHK(A,     "%02X");
    CHK(F,     "%02X");
    CHK(B,     "%02X");
    CHK(C,     "%02X");
    CHK(D,     "%02X");
    CHK(E,     "%02X");
    CHK(H,     "%02X");
    CHK(L,     "%02X");
    CHK(SP,    "%04X");
    CHK(fZ,    "%d");
    CHK(fN,    "%d");
    CHK(fH,    "%d");
    CHK(fC,    "%d");
#undef CHK

    /* memory writes */
    if (a->n_writes != b->n_writes) {
        fprintf(stderr, "  DIFF   writes: %s=%d  %s=%d\n",
                fa, a->n_writes, fb, b->n_writes);
    } else {
        for (int i = 0; i < a->n_writes; i++) {
            if (a->writes[i].addr != b->writes[i].addr ||
                a->writes[i].val  != b->writes[i].val) {
                fprintf(stderr,
                    "  DIFF   write[%d]: %s=%04X=%02X  %s=%04X=%02X\n",
                    i,
                    fa, a->writes[i].addr, a->writes[i].val,
                    fb, b->writes[i].addr, b->writes[i].val);
            }
        }
    }

    /* interrupts */
    if (a->interrupt_fired != b->interrupt_fired ||
        (a->interrupt_fired && a->int_vec != b->int_vec)) {
        fprintf(stderr,
            "  DIFF   interrupt: %s=%s(%02X)  %s=%s(%02X)\n",
            fa, a->interrupt_fired ? "YES" : "NO", a->int_vec,
            fb, b->interrupt_fired ? "YES" : "NO", b->int_vec);
    }
}

static bool records_equal(const TRec *a, const TRec *b) {
    if (a->cycle != b->cycle) return false;
    if (a->pc    != b->pc)    return false;
    if (a->A     != b->A)     return false;
    if (a->F     != b->F)     return false;
    if (a->B     != b->B)     return false;
    if (a->C     != b->C)     return false;
    if (a->D     != b->D)     return false;
    if (a->E     != b->E)     return false;
    if (a->H     != b->H)     return false;
    if (a->L     != b->L)     return false;
    if (a->SP    != b->SP)    return false;
    if (a->n_writes != b->n_writes) return false;
    for (int i = 0; i < a->n_writes; i++) {
        if (a->writes[i].addr != b->writes[i].addr) return false;
        if (a->writes[i].val  != b->writes[i].val)  return false;
    }
    return true;
}

/* ─── Main ───────────────────────────────────────────────────── */

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <trace_a> <trace_b>\n"
        "Options:\n"
        "  -n <count>   Stop after <count> instructions compared (default: all)\n"
        "  -s           Silent: only print the first divergence, no context\n"
        "  -h           Show this help\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *path_a = NULL, *path_b = NULL;
    uint64_t limit  = 0;  /* 0 = no limit */
    bool     silent = false;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "-s") == 0) { silent = true; }
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) { limit = (uint64_t)atol(argv[++i]); }
        else if (argv[i][0] != '-' && !path_a)  { path_a = argv[i]; }
        else if (argv[i][0] != '-' && !path_b)  { path_b = argv[i]; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }
    if (!path_a || !path_b) { usage(argv[0]); return 1; }

    FILE *fa = fopen(path_a, "r");
    FILE *fb = fopen(path_b, "r");
    if (!fa) { fprintf(stderr, "Cannot open: %s\n", path_a); return 1; }
    if (!fb) { fprintf(stderr, "Cannot open: %s\n", path_b); return 1; }

    TReader ra = { .f = fa };
    TReader rb = { .f = fb };
    Ring    cta = {0}, ctb = {0};

    uint64_t insn_count = 0;
    uint64_t diff_count = 0;
    const uint64_t MAX_DIFFS = 10;

    TRec ra_rec, rb_rec;
    bool ok_a, ok_b;

    while (1) {
        ok_a = tr_next(&ra, &ra_rec);
        ok_b = tr_next(&rb, &rb_rec);

        if (!ok_a && !ok_b) break;
        if (!ok_a) { fprintf(stderr, "[EOF] %s ended first at insn %llu\n", path_a, (unsigned long long)insn_count); break; }
        if (!ok_b) { fprintf(stderr, "[EOF] %s ended first at insn %llu\n", path_b, (unsigned long long)insn_count); break; }

        insn_count++;

        if (!records_equal(&ra_rec, &rb_rec)) {
            diff_count++;
            if (!silent) {
                ring_print(&cta, path_a);
                ring_print(&ctb, path_b);
            }
            print_diff(insn_count, &ra_rec, &rb_rec, path_a, path_b);
            if (diff_count >= MAX_DIFFS) {
                fprintf(stderr, "\n[STOP] %llu divergences found, stopping.\n",
                        (unsigned long long)diff_count);
                break;
            }
        }

        ring_push(&cta, &ra_rec);
        ring_push(&ctb, &rb_rec);

        if (limit > 0 && insn_count >= limit) break;
    }

    fclose(fa);
    fclose(fb);

    if (diff_count == 0)
        fprintf(stderr, "MATCH: %llu instructions compared, no divergence found.\n",
                (unsigned long long)insn_count);
    else
        fprintf(stderr, "\nSUMMARY: %llu divergences in %llu instructions.\n",
                (unsigned long long)diff_count,
                (unsigned long long)insn_count);

    return diff_count > 0 ? 1 : 0;
}
