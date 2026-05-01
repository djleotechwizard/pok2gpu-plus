/*
 * tools/trace_gen.c — offline profile-guided hot-trace compiler
 *
 * Usage:
 *   ./trace_gen <profile.bin> <rom.gb> [--output <path>] [--min-hits N]
 *                                      [--max-traces N] [--max-blocks N]
 *
 *   profile.bin  : written by gpu_batch_dump_profile():
 *                    uint32_t n_blocks
 *                    uint32_t counts[n_blocks]   (sorted-key order)
 *   rom.gb       : Pokémon Red ROM (to rebuild CFG)
 *   --output     : destination file (default: gpu/hot_traces.cuh)
 *   --min-hits N : minimum block hit count to trace (default 500)
 *   --max-traces N: max compiled traces (default 256)
 *   --max-blocks N: max blocks per trace (default 12)
 *
 * The tool rebuilds the same CFG + IR that gpu_batch_create uses,
 * finds the hottest straight-line superblocks, and emits CUDA C++
 * __device__ functions into gpu/hot_traces.cuh.  Recompile with
 * "make rl" to activate the traces.
 *
 * A trace is a chain of blocks.  Unconditional (FALLTHROUGH/JUMP)
 * transitions are followed greedily.  Conditional branches (EXIT_BRANCH)
 * are also extended via their fall-through arm: the taken path emits an
 * early exit (`goto trace_exit`) while the fall-through continues.
 * When the last block's fall-through loops back to the trace start, the
 * whole trace is wrapped in `for(;;)` (loop folding) — eliminating the
 * per-iteration hash-lookup overhead for tight GB loops.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "../analysis/cfg.h"
#include "../analysis/ir.h"

/* ── Defaults ──────────────────────────────────────────────── */
#define DEFAULT_OUTPUT      "gpu/hot_traces.cuh"
#define DEFAULT_MIN_HITS    500
#define DEFAULT_MAX_TRACES  256
#define DEFAULT_MAX_BLOCKS  12

/* Must match gpu/hot_traces.cuh */
#define TRACE_HASH_BITS  10
#define TRACE_HASH_SIZE  (1 << TRACE_HASH_BITS)

/* ── Trace representation ──────────────────────────────────── */
#define TRACE_MAX_BLOCKS 16

typedef struct {
    int      block_indices[TRACE_MAX_BLOCKS]; /* into sorted CFG block array */
    uint32_t start_key;                       /* (bank<<16)|entry of first block */
    int      n_blocks;
    int      max_tmps;
    int      total_fall_cycles;
    bool     is_loop;   /* last block's fall-through loops back to start_key */
} Trace;

/* ── Utilities ─────────────────────────────────────────────── */

static uint32_t block_key(const Block *b)
{
    return ((uint32_t)b->bank << 16) | (uint32_t)b->entry;
}

/* Find a block by key using binary search (blocks sorted by key) */
static int find_block(const Block *blocks, int n, uint32_t key)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t k = block_key(&blocks[mid]);
        if (k == key) return mid;
        if (k <  key) lo = mid + 1;
        else          hi = mid - 1;
    }
    return -1;
}

/* Load profile binary */
static uint32_t *load_profile(const char *path, int *out_n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open profile: %s\n", path); return NULL; }
    uint32_t n;
    if (fread(&n, sizeof(uint32_t), 1, f) != 1) { fclose(f); return NULL; }
    uint32_t *counts = (uint32_t *)calloc(n, sizeof(uint32_t));
    if (!counts) { fclose(f); return NULL; }
    if (fread(counts, sizeof(uint32_t), n, f) != n) {
        fprintf(stderr, "Profile truncated\n");
        free(counts); fclose(f); return NULL;
    }
    fclose(f);
    *out_n = (int)n;
    return counts;
}

/* Sort CFG blocks by key (they should already be sorted by cfg_build,
 * but sort again to match the GPU serialisation order exactly).       */
static int cmp_block_key(const void *a, const void *b)
{
    const Block *ba = (const Block *)a;
    const Block *bb = (const Block *)b;
    uint32_t ka = block_key(ba);
    uint32_t kb = block_key(bb);
    return (ka > kb) - (ka < kb);
}

/* ── Trace formation ───────────────────────────────────────── */

static Trace *form_traces(
    const Block    *blocks,   /* sorted by key */
    int             n_blocks,
    IRBlock       **ir,       /* [n_blocks]: lifted IR (may be NULL for CFG-miss) */
    const uint32_t *profile,  /* [n_blocks] hit counts */
    int             min_hits,
    int             max_traces,
    int             max_blocks_per_trace,
    int            *out_n_traces)
{
    Trace *traces = (Trace *)calloc((size_t)max_traces, sizeof(Trace));
    if (!traces) return NULL;
    int n_traces = 0;

    /* Boolean: is this block already the start of a trace? */
    bool *used_as_start = (bool *)calloc((size_t)n_blocks, sizeof(bool));
    if (!used_as_start) { free(traces); return NULL; }

    /* Sort block indices by hit count descending */
    int *order = (int *)malloc((size_t)n_blocks * sizeof(int));
    if (!order) { free(used_as_start); free(traces); return NULL; }
    for (int i = 0; i < n_blocks; i++) order[i] = i;
    /* Selection sort is fine for a few thousand blocks */
    for (int i = 0; i < n_blocks - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n_blocks; j++)
            if (profile[order[j]] > profile[order[best]]) best = j;
        if (best != i) { int tmp = order[i]; order[i] = order[best]; order[best] = tmp; }
    }

    for (int oi = 0; oi < n_blocks && n_traces < max_traces; oi++) {
        int start = order[oi];
        if (profile[start] < (uint32_t)min_hits) break; /* sorted desc, done */
        if (used_as_start[start]) continue;
        if (!ir[start]) continue; /* no IR for this block */

        Trace t;
        memset(&t, 0, sizeof(t));
        t.start_key = block_key(&blocks[start]);

        int cur = start;
        while (t.n_blocks < max_blocks_per_trace && t.n_blocks < TRACE_MAX_BLOCKS) {
            const Block  *blk = &blocks[cur];
            const IRBlock *ib = ir[cur];
            if (!ib) break;

            t.block_indices[t.n_blocks++] = cur;
            if (ib->n_tmps > t.max_tmps) t.max_tmps = ib->n_tmps;
            t.total_fall_cycles += ib->fall_cycles;

            /* Can we extend the trace?
             * FALLTHROUGH/JUMP  → follow succs[0] unconditionally.
             * BRANCH            → follow the fall-through arm:
             *     n_succs==2: succs[0]=taken, succs[1]=fall  (JP/JR cc)
             *     n_succs==1: succs[0]=fall                  (RET cc)
             * All other exits stop extension. */
            ExitType exit = blk->exit_type;
            uint16_t succ_addr;
            if (exit == EXIT_FALLTHROUGH || exit == EXIT_JUMP) {
                if (blk->n_succs < 1) break;
                succ_addr = blk->succs[0];
            } else if (exit == EXIT_BRANCH) {
                if      (blk->n_succs == 2) succ_addr = blk->succs[1]; /* fall-through */
                else if (blk->n_succs == 1) succ_addr = blk->succs[0]; /* RET cc fall  */
                else break;
            } else {
                break; /* CALL / RET / HALT / INDIRECT */
            }

            int      succ_bank = (succ_addr < 0x4000) ? 0 : blk->bank;
            uint32_t succ_key  = ((uint32_t)succ_bank << 16) | (uint32_t)succ_addr;

            /* Back-edge to trace start: self-loop detected, stop extending */
            if (succ_key == t.start_key) { t.is_loop = true; break; }

            int next = find_block(blocks, n_blocks, succ_key);
            if (next < 0 || !ir[next]) break;
            /* Don't enter another trace's start block */
            if (used_as_start[next]) break;

            cur = next;
        }

        /* Require at least 2 blocks for a useful trace */
        if (t.n_blocks < 2) continue;

        used_as_start[start] = true;
        traces[n_traces++] = t;
    }

    free(order);
    free(used_as_start);
    *out_n_traces = n_traces;
    return traces;
}

/* ── Code emission helpers ─────────────────────────────────── */

/* Emit a value read expression into buf (no newline) */
static void emit_read(char *buf, size_t sz, const IRVal *v)
{
    switch (v->kind) {
    case IRV_NONE:  snprintf(buf, sz, "0u");                  break;
    case IRV_A:     snprintf(buf, sz, "rA");                  break;
    case IRV_B:     snprintf(buf, sz, "rB");                  break;
    case IRV_C:     snprintf(buf, sz, "rC");                  break;
    case IRV_D:     snprintf(buf, sz, "rD");                  break;
    case IRV_E:     snprintf(buf, sz, "rE");                  break;
    case IRV_H:     snprintf(buf, sz, "rH");                  break;
    case IRV_L:     snprintf(buf, sz, "rL");                  break;
    case IRV_AF:    snprintf(buf, sz, "((uint32_t)(rA<<8u)|(rF&0xF0u))"); break;
    case IRV_BC:    snprintf(buf, sz, "((uint32_t)(rB<<8u)|rC)");         break;
    case IRV_DE:    snprintf(buf, sz, "((uint32_t)(rD<<8u)|rE)");         break;
    case IRV_HL:    snprintf(buf, sz, "((uint32_t)(rH<<8u)|rL)");         break;
    case IRV_SP:    snprintf(buf, sz, "(uint32_t)rSP");                   break;
    case IRV_ZF:    snprintf(buf, sz, "fZ");                  break;
    case IRV_NF:    snprintf(buf, sz, "fN");                  break;
    case IRV_HF:    snprintf(buf, sz, "fH");                  break;
    case IRV_CF:    snprintf(buf, sz, "fC");                  break;
    case IRV_IMM8:  snprintf(buf, sz, "%uu", (unsigned)v->u8);  break;
    case IRV_IMM16: snprintf(buf, sz, "%uu", (unsigned)v->u16); break;
    case IRV_TMP:   snprintf(buf, sz, "t%d", v->idx);            break;
    default:        snprintf(buf, sz, "0u");                  break;
    }
}

/* Emit a value read expression for use as a 16-bit memory address */
static void emit_addr(char *buf, size_t sz, const IRVal *v)
{
    switch (v->kind) {
    case IRV_A:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rA)"); break;
    case IRV_B:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rB)"); break;
    case IRV_C:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rC)"); break;
    case IRV_D:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rD)"); break;
    case IRV_E:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rE)"); break;
    case IRV_H:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rH)"); break;
    case IRV_L:     snprintf(buf, sz, "(uint16_t)(0xFF00u|rL)"); break;
    case IRV_BC:    snprintf(buf, sz, "(uint16_t)((uint32_t)(rB<<8u)|rC)"); break;
    case IRV_DE:    snprintf(buf, sz, "(uint16_t)((uint32_t)(rD<<8u)|rE)"); break;
    case IRV_HL:    snprintf(buf, sz, "(uint16_t)((uint32_t)(rH<<8u)|rL)"); break;
    case IRV_SP:    snprintf(buf, sz, "rSP");                                break;
    case IRV_IMM8:  snprintf(buf, sz, "(uint16_t)(0xFF00u|%uu)", (unsigned)v->u8);  break;
    case IRV_IMM16: snprintf(buf, sz, "(uint16_t)%uu", (unsigned)v->u16);            break;
    case IRV_TMP:   snprintf(buf, sz, "(t%d<0x100u?(uint16_t)(0xFF00u|t%d):(uint16_t)t%d)",
                             v->idx, v->idx, v->idx);
        break;
    default:        snprintf(buf, sz, "0u"); break;
    }
}

/* Emit a write statement: "dst = (expr);" — handles split pairs */
static void emit_write(FILE *f, const IRVal *dst, const char *expr)
{
    switch (dst->kind) {
    case IRV_A:  fprintf(f, "rA=(uint8_t)(%s);", expr);   break;
    case IRV_B:  fprintf(f, "rB=(uint8_t)(%s);", expr);   break;
    case IRV_C:  fprintf(f, "rC=(uint8_t)(%s);", expr);   break;
    case IRV_D:  fprintf(f, "rD=(uint8_t)(%s);", expr);   break;
    case IRV_E:  fprintf(f, "rE=(uint8_t)(%s);", expr);   break;
    case IRV_H:  fprintf(f, "rH=(uint8_t)(%s);", expr);   break;
    case IRV_L:  fprintf(f, "rL=(uint8_t)(%s);", expr);   break;
    case IRV_AF: fprintf(f, "{uint32_t _w=(%s);rA=(uint8_t)(_w>>8u);rF=(uint8_t)(_w&0xF0u);}", expr); break;
    case IRV_BC: fprintf(f, "{uint32_t _w=(%s);rB=(uint8_t)(_w>>8u);rC=(uint8_t)_w;}", expr); break;
    case IRV_DE: fprintf(f, "{uint32_t _w=(%s);rD=(uint8_t)(_w>>8u);rE=(uint8_t)_w;}", expr); break;
    case IRV_HL: fprintf(f, "{uint32_t _w=(%s);rH=(uint8_t)(_w>>8u);rL=(uint8_t)_w;}", expr); break;
    case IRV_SP: fprintf(f, "rSP=(uint16_t)(%s);", expr); break;
    case IRV_ZF: fprintf(f, "fZ=(uint32_t)(%s);", expr);  break;
    case IRV_NF: fprintf(f, "fN=(uint32_t)(%s);", expr);  break;
    case IRV_HF: fprintf(f, "fH=(uint32_t)(%s);", expr);  break;
    case IRV_CF: fprintf(f, "fC=(uint32_t)(%s);", expr);  break;
    case IRV_TMP: fprintf(f, "t%d=(uint32_t)(%s);", dst->idx, expr); break;
    default: break;
    }
}

/* Emit a per-block subsystem tick before every goto trace_exit.
 * Uses gpu_trace_subsys_tick (__noinline__) so NVCC compiles it once,
 * not inlined hundreds of times across all trace functions. */
static void emit_tick(FILE *f, int cyc)
{
    fprintf(f,
        " _total_cyc+=%d;"
        " gpu_trace_subsys_tick(s,ext_ram,rom,rom_size,ext_ram_size,%d);",
        cyc, cyc);
}

/* Emit one IR instruction as CUDA C++.
 * is_last_block : false → control-flow ops skipped (next block is known successor),
 *                 EXCEPT IR_BRANCH which emits the taken-path early exit.
 * blk_exit      : exit type of this block (needed for BRANCH variants).
 * loop_back     : true for the last block of a loop trace; the fall-through path
 *                 of IR_BRANCH/IR_JUMP loops back implicitly — suppress rPC= assign.
 * taken_cyc/fall_cyc: block cycle counts; injected before every goto trace_exit so
 *                 PPU/timer/DMA are ticked at block granularity inside the trace. */
static void emit_ir_insn(FILE *f, const IRInsn *ins,
                         bool is_last_block, ExitType blk_exit,
                         bool loop_back, int taken_cyc, int fall_cyc)
{
    char v0[64], v1[64], v2[64], a0[96], a1[96];
    emit_read(v0, sizeof(v0), &ins->src[0]);
    emit_read(v1, sizeof(v1), &ins->src[1]);
    emit_read(v2, sizeof(v2), &ins->src[2]);
    emit_addr(a0, sizeof(a0), &ins->src[0]);
    emit_addr(a1, sizeof(a1), &ins->src[1]);

    char expr[256];

    switch (ins->op) {
    case IR_NOP: break;

    case IR_MOV:
        emit_write(f, &ins->dst, v0);
        break;

    case IR_LOAD8:
        snprintf(expr, sizeof(expr),
            "gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,%s)", a0);
        emit_write(f, &ins->dst, expr);
        break;

    case IR_STORE8:
        fprintf(f, "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,%s,(uint8_t)%s);",
                a0, v1);
        break;

    case IR_ADD8:  snprintf(expr,sizeof(expr),"(%s+%s)&0xFFu",v0,v1);   emit_write(f,&ins->dst,expr); break;
    case IR_ADC8:  snprintf(expr,sizeof(expr),"(%s+%s+%s)&0xFFu",v0,v1,v2); emit_write(f,&ins->dst,expr); break;
    case IR_SUB8:  snprintf(expr,sizeof(expr),"((uint32_t)%s-%s)&0xFFu",v0,v1); emit_write(f,&ins->dst,expr); break;
    case IR_SBC8:  snprintf(expr,sizeof(expr),"((uint32_t)%s-%s-%s)&0xFFu",v0,v1,v2); emit_write(f,&ins->dst,expr); break;
    case IR_AND8:  snprintf(expr,sizeof(expr),"%s&%s",v0,v1);           emit_write(f,&ins->dst,expr); break;
    case IR_OR8:   snprintf(expr,sizeof(expr),"%s|%s",v0,v1);           emit_write(f,&ins->dst,expr); break;
    case IR_XOR8:  snprintf(expr,sizeof(expr),"%s^%s",v0,v1);           emit_write(f,&ins->dst,expr); break;
    case IR_NOT8:  snprintf(expr,sizeof(expr),"(~%s)&0xFFu",v0);        emit_write(f,&ins->dst,expr); break;
    case IR_INC8:  snprintf(expr,sizeof(expr),"(%s+1u)&0xFFu",v0);      emit_write(f,&ins->dst,expr); break;
    case IR_DEC8:  snprintf(expr,sizeof(expr),"((uint32_t)%s-1u)&0xFFu",v0); emit_write(f,&ins->dst,expr); break;

    case IR_ADD16:   snprintf(expr,sizeof(expr),"(%s+%s)&0xFFFFu",v0,v1); emit_write(f,&ins->dst,expr); break;
    case IR_INC16:   snprintf(expr,sizeof(expr),"(%s+1u)&0xFFFFu",v0);    emit_write(f,&ins->dst,expr); break;
    case IR_DEC16:   snprintf(expr,sizeof(expr),"((uint32_t)%s-1u)&0xFFFFu",v0); emit_write(f,&ins->dst,expr); break;
    case IR_ADD16_SP:
        snprintf(expr,sizeof(expr),
            "(uint32_t)((int32_t)(int16_t)rSP+(int8_t)(uint8_t)%s)&0xFFFFu",v0);
        emit_write(f,&ins->dst,expr); break;

    case IR_SHL8:  snprintf(expr,sizeof(expr),"(%s<<1u)&0xFFu",v0);     emit_write(f,&ins->dst,expr); break;
    case IR_SHR8:  snprintf(expr,sizeof(expr),"%s>>1u",v0);              emit_write(f,&ins->dst,expr); break;
    case IR_SAR8:  snprintf(expr,sizeof(expr),"(uint32_t)((int8_t)(uint8_t)%s>>1)&0xFFu",v0); emit_write(f,&ins->dst,expr); break;

    case IR_RLC8:
        fprintf(f, "{uint32_t _x=%s; ", v0);
        snprintf(expr,sizeof(expr),"((_x<<1u)|(_x>>7u))&0xFFu");
        emit_write(f,&ins->dst,expr); fprintf(f, "}"); break;
    case IR_RRC8:
        fprintf(f, "{uint32_t _x=%s; ", v0);
        snprintf(expr,sizeof(expr),"((_x>>1u)|(_x<<7u))&0xFFu");
        emit_write(f,&ins->dst,expr); fprintf(f, "}"); break;
    case IR_RL8:   snprintf(expr,sizeof(expr),"((%s<<1u)|fC)&0xFFu",v0); emit_write(f,&ins->dst,expr); break;
    case IR_RR8:   snprintf(expr,sizeof(expr),"((%s>>1u)|(fC<<7u))&0xFFu",v0); emit_write(f,&ins->dst,expr); break;
    case IR_SWAP8:
        fprintf(f, "{uint32_t _x=%s; ", v0);
        snprintf(expr,sizeof(expr),"((_x&0xFu)<<4u)|((_x>>4u)&0xFu)");
        emit_write(f,&ins->dst,expr); fprintf(f, "}"); break;
    case IR_BIT8:  snprintf(expr,sizeof(expr),"(%s>>(%s&7u))&1u",v0,v1);       emit_write(f,&ins->dst,expr); break;
    case IR_RES8:  snprintf(expr,sizeof(expr),"%s&~(1u<<(%s&7u))",v0,v1);       emit_write(f,&ins->dst,expr); break;
    case IR_SET8:  snprintf(expr,sizeof(expr),"%s|(1u<<(%s&7u))",v0,v1);        emit_write(f,&ins->dst,expr); break;

    case IR_ZFLAG:     snprintf(expr,sizeof(expr),"(%s==0u)?1u:0u",v0);                   emit_write(f,&ins->dst,expr); break;
    case IR_HFLAG_ADD: snprintf(expr,sizeof(expr),"((%s&0xFu)+(%s&0xFu)+%s)>0xFu?1u:0u",v0,v1,v2); emit_write(f,&ins->dst,expr); break;
    case IR_HFLAG_SUB: snprintf(expr,sizeof(expr),"((int)(%s&0xFu)-(int)(%s&0xFu)-(int)%s)<0?1u:0u",v0,v1,v2); emit_write(f,&ins->dst,expr); break;
    case IR_CFLAG_ADD: snprintf(expr,sizeof(expr),"(%s+%s+%s)>0xFFu?1u:0u",v0,v1,v2);    emit_write(f,&ins->dst,expr); break;
    case IR_CFLAG_SUB: snprintf(expr,sizeof(expr),"((int)%s-(int)%s-(int)%s)<0?1u:0u",v0,v1,v2); emit_write(f,&ins->dst,expr); break;
    case IR_HFLAG_A16: snprintf(expr,sizeof(expr),"((%s&0xFFFu)+(%s&0xFFFu))>0xFFFu?1u:0u",v0,v1); emit_write(f,&ins->dst,expr); break;
    case IR_CFLAG_A16: snprintf(expr,sizeof(expr),"((uint32_t)%s+%s)>0xFFFFu?1u:0u",v0,v1); emit_write(f,&ins->dst,expr); break;
    case IR_HFLAG_SP:
        snprintf(expr,sizeof(expr),"((rSP&0xFu)+((uint32_t)((uint8_t)(int8_t)(uint8_t)%s)&0xFu))>0xFu?1u:0u",v0);
        emit_write(f,&ins->dst,expr); break;
    case IR_CFLAG_SP:
        snprintf(expr,sizeof(expr),"((rSP&0xFFu)+(uint32_t)((uint8_t)(int8_t)(uint8_t)%s))>0xFFu?1u:0u",v0);
        emit_write(f,&ins->dst,expr); break;

    case IR_PUSH16:
        fprintf(f,
            "{uint32_t _pv=%s;rSP=(uint16_t)(rSP-2u);"
            "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u),(uint8_t)(_pv>>8u));"
            "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP,(uint8_t)_pv);}",
            v0);
        break;
    case IR_POP16:
        fprintf(f,
            "{uint8_t _lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP);"
            "uint8_t _hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u));"
            "rSP=(uint16_t)(rSP+2u);");
        snprintf(expr,sizeof(expr),"(uint32_t)(((uint32_t)_hi<<8u)|_lo)");
        emit_write(f,&ins->dst,expr);
        fprintf(f, "}");
        break;

    /* ── Control flow ──────────────────────────────────────────── */
    case IR_JUMP:
        if (is_last_block) {
            if (loop_back) {
                fprintf(f, "/* loop back (unconditional jump) */");
            } else {
                fprintf(f, "rPC=(uint16_t)%s;", v0);
                emit_tick(f, fall_cyc);
                fprintf(f, " goto trace_exit;");
            }
        }
        /* else: interior unconditional jump — fall through to next trace block */
        break;

    case IR_BRANCH:
        if (!is_last_block) {
            /* Interior block: emit early-exit for the taken path only.
             * The fall-through arm continues into the next trace block. */
            if (ins->src[1].kind == IRV_IMM16 && ins->src[1].u16 == 0xFFFFu) {
                /* Interior RET cc */
                fprintf(f,
                    "if(%s){"
                    "uint8_t _lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP);"
                    "uint8_t _hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u));"
                    "rSP=(uint16_t)(rSP+2u);"
                    "rPC=(uint16_t)(((uint32_t)_hi<<8u)|_lo);",
                    v0);
                emit_tick(f, taken_cyc);
                fprintf(f, " goto trace_exit;}");
            } else if (blk_exit == EXIT_CALL_CC) {
                /* Interior CALL cc */
                fprintf(f,
                    "if(%s){"
                    "uint16_t _ret=(uint16_t)%s;rSP=(uint16_t)(rSP-2u);"
                    "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u),(uint8_t)(_ret>>8u));"
                    "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP,(uint8_t)_ret);"
                    "rPC=(uint16_t)%s;",
                    v0, v2, v1);
                emit_tick(f, taken_cyc);
                fprintf(f, " goto trace_exit;}");
            } else {
                /* Interior normal branch */
                fprintf(f, "if(%s){rPC=(uint16_t)%s;", v0, v1);
                emit_tick(f, taken_cyc);
                fprintf(f, " goto trace_exit;}");
            }
        } else { /* is_last_block */
            if (ins->src[1].kind == IRV_IMM16 && ins->src[1].u16 == 0xFFFFu) {
                /* Last RET cc */
                if (loop_back) {
                    /* Fall-through = back edge: no rPC= needed, for(;;) continues */
                    fprintf(f,
                        "if(%s){"
                        "uint8_t _lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP);"
                        "uint8_t _hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u));"
                        "rSP=(uint16_t)(rSP+2u);"
                        "rPC=(uint16_t)(((uint32_t)_hi<<8u)|_lo);",
                        v0);
                    emit_tick(f, taken_cyc);
                    fprintf(f, " goto trace_exit;}");
                } else {
                    fprintf(f,
                        "if(%s){"
                        "uint8_t _lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP);"
                        "uint8_t _hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u));"
                        "rSP=(uint16_t)(rSP+2u);"
                        "rPC=(uint16_t)(((uint32_t)_hi<<8u)|_lo);",
                        v0);
                    emit_tick(f, taken_cyc);
                    fprintf(f, " goto trace_exit;}"
                        "rPC=(uint16_t)%s;",
                        v2);
                }
            } else if (blk_exit == EXIT_CALL_CC) {
                /* Last CALL cc */
                fprintf(f,
                    "if(%s){"
                    "uint16_t _ret=(uint16_t)%s;rSP=(uint16_t)(rSP-2u);"
                    "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u),(uint8_t)(_ret>>8u));"
                    "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP,(uint8_t)_ret);"
                    "rPC=(uint16_t)%s;",
                    v0, v2, v1);
                emit_tick(f, taken_cyc);
                fprintf(f, " goto trace_exit;}"
                    "rPC=(uint16_t)%s;",
                    v2);
            } else {
                /* Last normal branch */
                if (loop_back) {
                    /* Fall-through = back edge */
                    fprintf(f, "if(%s){rPC=(uint16_t)%s;", v0, v1);
                    emit_tick(f, taken_cyc);
                    fprintf(f, " goto trace_exit;}");
                } else {
                    fprintf(f, "if(%s){rPC=(uint16_t)%s;", v0, v1);
                    emit_tick(f, taken_cyc);
                    fprintf(f, " goto trace_exit;}"
                        "rPC=(uint16_t)%s;",
                        v2);
                }
            }
        }
        break;

    case IR_CALL:
        if (is_last_block) {
            fprintf(f,
                "{uint16_t _ret=(uint16_t)%s;rSP=(uint16_t)(rSP-2u);"
                "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u),(uint8_t)(_ret>>8u));"
                "gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP,(uint8_t)_ret);"
                "rPC=(uint16_t)%s;",
                v1, v0);
            emit_tick(f, taken_cyc);
            fprintf(f, " goto trace_exit;}");
        }
        break;

    case IR_RET:
        if (is_last_block) {
            fprintf(f,
                "{uint8_t _lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,rSP);"
                "uint8_t _hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(rSP+1u));"
                "rSP=(uint16_t)(rSP+2u);"
                "rPC=(uint16_t)(((uint32_t)_hi<<8u)|_lo);");
            emit_tick(f, taken_cyc);
            fprintf(f, " goto trace_exit;}");
        }
        break;

    case IR_HALT:
        emit_tick(f, fall_cyc);
        fprintf(f, " s->halted=1; goto trace_exit;");
        break;

    case IR_DAA:
        fprintf(f,
            "{uint8_t _a=rA,_cr=0;int _sc=0;"
            "if(!fN){if(fH|((_a&0xFu)>9u)){_cr|=0x06u;}"
            "if(fC|_a>0x99u){_cr|=0x60u;_sc=1;}_a=(uint8_t)(_a+_cr);}"
            "else{if(fH)_cr|=0x06u;if(fC){_cr|=0x60u;_sc=1;}_a=(uint8_t)(_a-_cr);}"
            "fZ=(_a==0u)?1u:0u;fH=0u;if(_sc)fC=1u;rA=_a;}");
        break;

    case IR_IME_SET: fprintf(f, "s->ime_pending=1;"); break;
    case IR_IME_CLR: fprintf(f, "s->IME=0;s->ime_pending=0;"); break;

    default: break;
    }
    fprintf(f, "\n");
}

/* ── Emit one trace function ───────────────────────────────── */

static void emit_trace_function(FILE *f, int trace_id, const Trace *t,
                                 const Block *blocks, IRBlock **ir)
{
    /* Determine the exit PC for the common (fall-through) path.
     * For the last block:  EXIT_FALLTHROUGH → succs[0]
     *                      EXIT_JUMP        → succs[0]
     *                      EXIT_BRANCH etc  → handled by IR_BRANCH emission
     */
    const Block  *last_blk = &blocks[t->block_indices[t->n_blocks - 1]];

    uint16_t exit_pc = 0;
    if (last_blk->n_succs > 0 &&
        (last_blk->exit_type == EXIT_FALLTHROUGH ||
         last_blk->exit_type == EXIT_JUMP))
        exit_pc = last_blk->succs[0];

    /* Decide tmp count: round up to nearest template bucket */
    int nt = t->max_tmps;
    if (nt < 1) nt = 1;

    fprintf(f,
        "/* trace %d: start=0x%04X bank=%d  %d blocks  fall_cycles=%d */\n"
        "__device__ static int jit_trace_%03d(\n"
        "    GBStateGPU *s, uint8_t *vram, uint8_t *wram, uint8_t *ext_ram, const uint8_t *rom,\n"
        "    uint32_t rom_size, uint32_t ext_ram_size)\n"
        "{\n"
        "    /* Load hot registers into locals (stay in CUDA registers) */\n"
        "    uint8_t  rA=s->A, rB=s->B, rC=s->C, rD=s->D;\n"
        "    uint8_t  rE=s->E, rH=s->H, rL=s->L, rF=s->F;\n"
        "    uint16_t rSP=s->SP;\n"
        "    uint32_t fZ=(rF>>7u)&1u, fN=(rF>>6u)&1u;\n"
        "    uint32_t fH=(rF>>5u)&1u, fC=(rF>>4u)&1u;\n",
        trace_id,
        (unsigned)(t->start_key & 0xFFFFu),
        (int)(t->start_key >> 16),
        t->n_blocks,
        t->total_fall_cycles,
        trace_id);

    /* Declare temporaries */
    if (nt > 0) {
        fprintf(f, "    uint32_t");
        for (int i = 0; i < nt; i++)
            fprintf(f, "%s t%d=0u", i ? "," : " ", i);
        fprintf(f, ";\n");
    }

    fprintf(f,
        "    int _total_cyc = 0;\n"
        "    uint16_t rPC = 0u; /* set by terminal instruction */\n"
        "    (void)rPC;\n\n");

    /* Loop-folded traces wrap all blocks in for(;;).
     * The back-edge fall-through lets the loop continue naturally;
     * all other exits use `goto trace_exit` to escape. */
    if (t->is_loop)
        fprintf(f, "    for (;;) { /* loop trace */\n");

    /* Emit each block */
    for (int bi = 0; bi < t->n_blocks; bi++) {
        int idx = t->block_indices[bi];
        const Block  *blk = &blocks[idx];
        const IRBlock *ib  = ir[idx];
        bool is_last   = (bi == t->n_blocks - 1);
        bool loop_back = is_last && t->is_loop;

        fprintf(f,
            "    /* ── block %d: bank=%d entry=0x%04X %s%s ── */\n",
            bi, blk->bank, blk->entry,
            is_last ? "LAST" : "interior",
            loop_back ? " (loop-back)" : "");

        for (int ii = 0; ii < ib->n_insns; ii++) {
            fprintf(f, "    ");
            emit_ir_insn(f, &ib->insns[ii], is_last, blk->exit_type, loop_back,
                         ib->taken_cycles, ib->fall_cycles);
        }

        /* Fall-through path: accumulate cycles and tick subsystems per block.
         * Early exits (goto trace_exit) from emit_ir_insn already tick inline. */
        fprintf(f,
            "    _total_cyc += %d;\n"
            "    gpu_trace_subsys_tick(s, ext_ram, rom, rom_size, ext_ram_size, %d);\n",
            ib->fall_cycles, ib->fall_cycles);

        /* Interrupt check after interior blocks and loop-back last block. */
        if (!is_last) {
            /* Interior: exit to next block's entry so interrupt is handled before it runs. */
            const Block *next_blk = &blocks[t->block_indices[bi + 1]];
            fprintf(f,
                "    if (s->io[0x0F] & s->IE & 0x1Fu) { rPC = 0x%04Xu; goto trace_exit; }\n",
                (unsigned)next_blk->entry);
        } else if (loop_back) {
            /* Loop-back last block: exit to trace entry so interrupt runs next. */
            fprintf(f,
                "    if (s->io[0x0F] & s->IE & 0x1Fu) { rPC = 0x%04Xu; goto trace_exit; }\n",
                (unsigned)(t->start_key & 0xFFFFu));
        }

        fprintf(f, "\n");
    }

    if (t->is_loop)
        fprintf(f,
            "    if (_total_cyc >= 65536) { rPC = 0x%04Xu; goto trace_exit; } /* yield: resume at loop top */\n"
            "    } /* end loop trace */\n",
            (unsigned)(t->start_key & 0xFFFFu));

    /* Fall-through exit PC (only for non-loop EXIT_FALLTHROUGH/EXIT_JUMP) */
    if (!t->is_loop &&
        (last_blk->exit_type == EXIT_FALLTHROUGH ||
         last_blk->exit_type == EXIT_JUMP))
        fprintf(f, "    rPC = 0x%04Xu;\n", (unsigned)exit_pc);

    fprintf(f,
        "trace_exit:\n"
        "    rF = (uint8_t)((fZ<<7u)|(fN<<6u)|(fH<<5u)|(fC<<4u));\n"
        "    s->A=rA; s->B=rB; s->C=rC; s->D=rD;\n"
        "    s->E=rE; s->H=rH; s->L=rL; s->F=rF;\n"
        "    s->SP=rSP; s->PC=rPC;\n"
        "    s->cycles += (uint64_t)_total_cyc;\n"
        "    return _total_cyc;\n"
        "}\n\n");
}

/* ── Emit full hot_traces.cuh ─────────────────────────────── */

static void emit_header(FILE *f, const Trace *traces, int n_traces,
                         const Block *blocks, IRBlock **ir)
{
    fprintf(f,
        "/* gpu/hot_traces.cuh — JIT-compiled hot-trace dispatch\n"
        " * AUTO-GENERATED by tools/trace_gen — do not edit.\n"
        " * Regenerate: ./trace_gen <profile.bin> <rom.gb>\n"
        " * Then recompile: make rl\n"
        " */\n"
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        "/* Forward declaration for the combined per-block tick (defined in batch_gpu.cu).\n"
        " * __noinline__ prevents NVCC from expanding it hundreds of times across traces. */\n"
        "__device__ __noinline__ static void gpu_trace_subsys_tick(\n"
        "    GBStateGPU *s, uint8_t *ext_ram, const uint8_t *rom,\n"
        "    uint32_t rom_size, uint32_t ext_ram_size, int cyc);\n\n"
        "/* ── Trace dispatch hash table (constant memory) ─── */\n"
        "#define TRACE_HASH_BITS  10\n"
        "#define TRACE_HASH_SIZE  (1 << TRACE_HASH_BITS)\n"
        "#define TRACE_HASH_EMPTY 0xFFFFFFFFu\n\n"
        "typedef struct {\n"
        "    uint32_t key;\n"
        "    int32_t  idx;\n"
        "} TraceHashEntry;\n\n"
        "__constant__ static TraceHashEntry c_trace_hash[TRACE_HASH_SIZE];\n\n"
        "__device__ __forceinline__ static int trace_lookup(uint32_t key)\n"
        "{\n"
        "    uint32_t h = (key * 2654435761u) >> (32 - TRACE_HASH_BITS);\n"
        "#pragma unroll 4\n"
        "    for (int i = 0; i < 8; i++) {\n"
        "        uint32_t k = c_trace_hash[h].key;\n"
        "        if (k == key)              return c_trace_hash[h].idx;\n"
        "        if (k == TRACE_HASH_EMPTY) return -1;\n"
        "        h = (h + 1) & (TRACE_HASH_SIZE - 1);\n"
        "    }\n"
        "    return -1;\n"
        "}\n\n"
        "#define N_JIT_TRACES %d\n\n",
        n_traces);

    /* Forward declarations */
    for (int i = 0; i < n_traces; i++)
        fprintf(f,
            "__device__ static int jit_trace_%03d(\n"
            "    GBStateGPU*, uint8_t*, uint8_t*, uint8_t*, const uint8_t*, uint32_t, uint32_t);\n",
            i);
    fprintf(f, "\n");

    /* Trace executor */
    fprintf(f,
        "__device__ static int jit_exec_trace(\n"
        "    int trace_id, GBStateGPU *s, uint8_t *vram, uint8_t *wram, uint8_t *ext_ram,\n"
        "    const uint8_t *rom, uint32_t rom_size, uint32_t ext_ram_size)\n"
        "{\n"
        "    switch (trace_id) {\n");
    for (int i = 0; i < n_traces; i++)
        fprintf(f,
            "    case %d: return jit_trace_%03d(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size);\n",
            i, i);
    fprintf(f,
        "    default: return 0;\n"
        "    }\n"
        "}\n\n");

    /* Hash table build (global, uploaded via trace_hash_upload) */
    fprintf(f, "static TraceHashEntry g_jit_trace_hash[TRACE_HASH_SIZE] = {\n");
    /* Build hash table entries */
    typedef struct { uint32_t key; int32_t idx; } Slot;
    Slot *ht = (Slot *)calloc(TRACE_HASH_SIZE, sizeof(Slot));
    if (!ht) { fprintf(stderr, "OOM building hash table\n"); return; }
    for (int i = 0; i < TRACE_HASH_SIZE; i++) { ht[i].key = 0xFFFFFFFFu; ht[i].idx = -1; }
    for (int i = 0; i < n_traces; i++) {
        uint32_t k = traces[i].start_key;
        uint32_t h = (k * 2654435761u) >> (32 - TRACE_HASH_BITS);
        while (ht[h].key != 0xFFFFFFFFu) h = (h + 1) & (TRACE_HASH_SIZE - 1);
        ht[h].key = k;
        ht[h].idx = i;
    }
    for (int i = 0; i < TRACE_HASH_SIZE; i++) {
        if (ht[i].key == 0xFFFFFFFFu)
            fprintf(f, "    {0xFFFFFFFFu,-1},\n");
        else
            fprintf(f, "    {0x%08Xu,%d},\n", ht[i].key, ht[i].idx);
    }
    free(ht);
    fprintf(f, "};\n\n");

    /* trace_hash_upload */
    fprintf(f,
        "static inline void trace_hash_upload(void)\n"
        "{\n"
        "    cudaMemcpyToSymbol(c_trace_hash, g_jit_trace_hash,\n"
        "                       TRACE_HASH_SIZE * sizeof(TraceHashEntry));\n"
        "}\n\n");

    /* Trace function bodies */
    for (int i = 0; i < n_traces; i++)
        emit_trace_function(f, i, &traces[i], blocks, ir);
}

/* ── main ──────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *profile_path = NULL;
    const char *rom_path     = NULL;
    const char *output_path  = DEFAULT_OUTPUT;
    int         min_hits     = DEFAULT_MIN_HITS;
    int         max_traces   = DEFAULT_MAX_TRACES;
    int         max_blk      = DEFAULT_MAX_BLOCKS;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--output")    && i+1<argc) output_path = argv[++i];
        else if (!strcmp(argv[i], "--min-hits")  && i+1<argc) min_hits    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-traces")&& i+1<argc) max_traces  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-blocks")&& i+1<argc) max_blk     = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !profile_path) profile_path = argv[i];
        else if (argv[i][0] != '-' && !rom_path)     rom_path     = argv[i];
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); return 1; }
    }
    if (!profile_path || !rom_path) {
        fprintf(stderr,
            "Usage: trace_gen <profile.bin> <rom.gb> [--output F]"
            " [--min-hits N] [--max-traces N] [--max-blocks N]\n");
        return 1;
    }

    /* Load ROM */
    FILE *rf = fopen(rom_path, "rb");
    if (!rf) { fprintf(stderr, "Cannot open ROM: %s\n", rom_path); return 1; }
    fseek(rf, 0, SEEK_END);
    long rom_sz = ftell(rf);
    rewind(rf);
    uint8_t *rom = (uint8_t *)malloc((size_t)rom_sz);
    if (!rom) { fclose(rf); return 1; }
    if (fread(rom, 1, (size_t)rom_sz, rf) != (size_t)rom_sz) {
        fprintf(stderr, "ROM read error\n"); free(rom); fclose(rf); return 1;
    }
    fclose(rf);
    fprintf(stderr, "ROM: %ld bytes\n", rom_sz);

    /* Build CFG */
    fprintf(stderr, "Building CFG...\n");
    CFG *cfg = cfg_build(rom, (uint32_t)rom_sz);
    if (!cfg) { fprintf(stderr, "cfg_build failed\n"); free(rom); return 1; }
    fprintf(stderr, "CFG: %d blocks\n", cfg->n_blocks);

    /* Sort blocks by key (must match GPU serialisation order) */
    qsort(cfg->blocks, (size_t)cfg->n_blocks, sizeof(Block), cmp_block_key);

    /* Load profile */
    int profile_n = 0;
    uint32_t *profile = load_profile(profile_path, &profile_n);
    if (!profile) { cfg_free(cfg); free(rom); return 1; }
    fprintf(stderr, "Profile: %d block counts loaded\n", profile_n);

    if (profile_n != cfg->n_blocks) {
        fprintf(stderr,
            "WARNING: profile has %d blocks but CFG has %d. "
            "Profile was generated with a different ROM build.\n",
            profile_n, cfg->n_blocks);
        /* Truncate to smaller of the two */
        if (profile_n > cfg->n_blocks) profile_n = cfg->n_blocks;
    }

    /* Lift IR for all blocks (lazy: only hot ones will be used) */
    fprintf(stderr, "Lifting IR for hot blocks...\n");
    IRBlock **ir = (IRBlock **)calloc((size_t)cfg->n_blocks, sizeof(IRBlock *));
    if (!ir) { free(profile); cfg_free(cfg); free(rom); return 1; }
    uint64_t total_hits = 0;
    int hot_count = 0;
    for (int i = 0; i < profile_n; i++) {
        total_hits += profile[i];
        if (profile[i] >= (uint32_t)min_hits) {
            ir[i] = ir_lift_block(&cfg->blocks[i]);
            hot_count++;
        }
    }
    fprintf(stderr, "Hot blocks (>= %d hits): %d  total hits: %llu\n",
            min_hits, hot_count, (unsigned long long)total_hits);

    /* Form traces */
    int n_traces = 0;
    Trace *traces = form_traces(cfg->blocks, cfg->n_blocks,
                                ir, profile,
                                min_hits, max_traces, max_blk,
                                &n_traces);
    if (!traces) { fprintf(stderr, "form_traces OOM\n"); return 1; }
    fprintf(stderr, "Traces formed: %d\n", n_traces);

    /* Print summary of top traces */
    fprintf(stderr, "Top traces:\n");
    for (int i = 0; i < n_traces && i < 10; i++) {
        const Block *b0 = &cfg->blocks[traces[i].block_indices[0]];
        uint32_t hits = profile[traces[i].block_indices[0]];
        fprintf(stderr,
            "  [%3d] bank=%d entry=0x%04X  %d blocks  %d cycles  %u hits%s\n",
            i, b0->bank, b0->entry,
            traces[i].n_blocks,
            traces[i].total_fall_cycles,
            hits,
            traces[i].is_loop ? "  [LOOP]" : "");
    }

    /* Emit output */
    FILE *out = fopen(output_path, "w");
    if (!out) { fprintf(stderr, "Cannot write: %s\n", output_path); return 1; }
    emit_header(out, traces, n_traces, cfg->blocks, ir);
    fclose(out);
    fprintf(stderr, "Written: %s  (%d traces)\n", output_path, n_traces);
    fprintf(stderr, "Recompile with: make rl\n");

    /* Cleanup */
    for (int i = 0; i < cfg->n_blocks; i++)
        if (ir[i]) ir_free_block(ir[i]);
    free(ir);
    free(traces);
    free(profile);
    cfg_free(cfg);
    free(rom);
    return 0;
}
