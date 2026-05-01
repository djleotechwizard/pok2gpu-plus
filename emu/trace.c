#include <string.h>
#include "gb.h"

void tracer_init(Tracer *t, const char *path) {
    memset(t, 0, sizeof(*t));
    if (path) {
        t->file = fopen(path, "w");
        if (!t->file) {
            fprintf(stderr, "Cannot open trace file: %s\n", path);
            return;
        }
    }
    t->enabled = (t->file != NULL);
    t->entry_count = 0;
}

void tracer_close(Tracer *t) {
    if (t->file) { fclose(t->file); t->file = NULL; }
    t->enabled = false;
}

/*
 * Trace format (one line per instruction):
 * T:000000001234 PC:0100 OP:00        A:01 F:B0 B:00 C:13 ...
 * Followed by optional W:ADDR=VAL and INT:VEC lines.
 */
void tracer_emit(Tracer *t, const TraceEntry *e) {
    if (!t->enabled || !t->file) return;

    int fZ = (e->F & FLAG_Z) ? 1 : 0;
    int fN = (e->F & FLAG_N) ? 1 : 0;
    int fH = (e->F & FLAG_H) ? 1 : 0;
    int fC = (e->F & FLAG_C) ? 1 : 0;

    char opbuf[16];
    if (e->opcode_len == 1)
        snprintf(opbuf, sizeof(opbuf), "%02X      ", e->opcode[0]);
    else if (e->opcode_len == 2)
        snprintf(opbuf, sizeof(opbuf), "%02X %02X   ", e->opcode[0], e->opcode[1]);
    else
        snprintf(opbuf, sizeof(opbuf), "%02X %02X %02X", e->opcode[0], e->opcode[1], e->opcode[2]);

    if (e->interrupt_fired)
        fprintf(t->file, "INT:%02X\n", e->interrupt_vector);

    fprintf(t->file,
        "T:%012llu PC:%04X OP:%s  "
        "A:%02X F:%02X B:%02X C:%02X D:%02X E:%02X H:%02X L:%02X SP:%04X  "
        "Z:%d N:%d H:%d C:%d\n",
        (unsigned long long)e->cycle,
        e->pc, opbuf,
        e->A, e->F, e->B, e->C, e->D, e->E, e->H, e->L, e->SP,
        fZ, fN, fH, fC);

    for (int i = 0; i < e->mem_write_count; i++)
        fprintf(t->file, "  W:%04X=%02X\n",
            e->mem_writes[i].addr, e->mem_writes[i].value);

    t->entry_count++;
    if ((t->entry_count & 0xFFF) == 0) fflush(t->file);
}
