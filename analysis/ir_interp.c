#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ir_interp.h"

/* ============================================================
 * Value accessors
 * ============================================================ */

static uint32_t get_val(const IRVal *v, const GB *gb, const uint32_t *tmps)
{
    switch (v->kind) {
        case IRV_NONE:  return 0;
        case IRV_A:     return gb->cpu.A;
        case IRV_B:     return gb->cpu.B;
        case IRV_C:     return gb->cpu.C;
        case IRV_D:     return gb->cpu.D;
        case IRV_E:     return gb->cpu.E;
        case IRV_H:     return gb->cpu.H;
        case IRV_L:     return gb->cpu.L;
        case IRV_AF:    return cpu_AF(&gb->cpu);
        case IRV_BC:    return cpu_BC(&gb->cpu);
        case IRV_DE:    return cpu_DE(&gb->cpu);
        case IRV_HL:    return cpu_HL(&gb->cpu);
        case IRV_SP:    return gb->cpu.SP;
        case IRV_ZF:    return GET_Z(&gb->cpu) ? 1u : 0u;
        case IRV_NF:    return GET_N(&gb->cpu) ? 1u : 0u;
        case IRV_HF:    return GET_H(&gb->cpu) ? 1u : 0u;
        case IRV_CF:    return GET_C(&gb->cpu) ? 1u : 0u;
        case IRV_IMM8:  return v->u8;
        case IRV_IMM16: return v->u16;
        case IRV_TMP:   return tmps[v->idx];
        default:        return 0;
    }
}

static void set_val(const IRVal *v, GB *gb, uint32_t *tmps, uint32_t val)
{
    switch (v->kind) {
        case IRV_A:  gb->cpu.A = (uint8_t)val;  break;
        case IRV_B:  gb->cpu.B = (uint8_t)val;  break;
        case IRV_C:  gb->cpu.C = (uint8_t)val;  break;
        case IRV_D:  gb->cpu.D = (uint8_t)val;  break;
        case IRV_E:  gb->cpu.E = (uint8_t)val;  break;
        case IRV_H:  gb->cpu.H = (uint8_t)val;  break;
        case IRV_L:  gb->cpu.L = (uint8_t)val;  break;
        case IRV_AF: cpu_set_AF(&gb->cpu, (uint16_t)val); break;
        case IRV_BC: cpu_set_BC(&gb->cpu, (uint16_t)val); break;
        case IRV_DE: cpu_set_DE(&gb->cpu, (uint16_t)val); break;
        case IRV_HL: cpu_set_HL(&gb->cpu, (uint16_t)val); break;
        case IRV_SP: gb->cpu.SP = (uint16_t)val; break;
        case IRV_ZF: SET_Z(&gb->cpu, val); break;
        case IRV_NF: SET_N(&gb->cpu, val); break;
        case IRV_HF: SET_H(&gb->cpu, val); break;
        case IRV_CF: SET_C(&gb->cpu, val); break;
        case IRV_TMP: tmps[v->idx] = val; break;
        default: break;
    }
}

/* Resolve a LOAD8/STORE8 address value.
 * 8-bit register kinds (A..L) and TMPs with values < 0x100 are LDH-style
 * (0xFF00-based) — all other SM83 memory ops use 16-bit address sources. */
static uint16_t addr_val(const IRVal *v, const GB *gb, const uint32_t *tmps)
{
    switch (v->kind) {
        /* 8-bit registers used as address → LDH 0xFF00+r */
        case IRV_A: return (uint16_t)(0xFF00u | gb->cpu.A);
        case IRV_B: return (uint16_t)(0xFF00u | gb->cpu.B);
        case IRV_C: return (uint16_t)(0xFF00u | gb->cpu.C);
        case IRV_D: return (uint16_t)(0xFF00u | gb->cpu.D);
        case IRV_E: return (uint16_t)(0xFF00u | gb->cpu.E);
        case IRV_H: return (uint16_t)(0xFF00u | gb->cpu.H);
        case IRV_L: return (uint16_t)(0xFF00u | gb->cpu.L);
        /* 16-bit registers / immediates → direct address */
        case IRV_BC:    return cpu_BC(&gb->cpu);
        case IRV_DE:    return cpu_DE(&gb->cpu);
        case IRV_HL:    return cpu_HL(&gb->cpu);
        case IRV_SP:    return gb->cpu.SP;
        case IRV_IMM16: return v->u16;
        case IRV_IMM8:  return (uint16_t)(0xFF00u | v->u8);
        case IRV_TMP: {
            uint32_t x = tmps[v->idx];
            /* LDH (C),A stores C (8-bit) in a tmp via OR8; detect by range */
            return (x < 0x100) ? (uint16_t)(0xFF00u | x) : (uint16_t)x;
        }
        default: return 0;
    }
}

/* DAA correction (mirrors the SM83 spec) */
static uint8_t exec_daa(CPU *cpu)
{
    uint8_t a = cpu->A;
    uint8_t corr = 0;
    int set_c = 0;

    if (!GET_N(cpu)) {
        if (GET_H(cpu) || (a & 0x0F) > 9)  corr |= 0x06;
        if (GET_C(cpu) || a > 0x99)         { corr |= 0x60; set_c = 1; }
        a = (uint8_t)(a + corr);
    } else {
        if (GET_H(cpu)) corr |= 0x06;
        if (GET_C(cpu)) { corr |= 0x60; set_c = 1; }
        a = (uint8_t)(a - corr);
    }
    SET_Z(cpu, a == 0);
    SET_H(cpu, 0);
    if (set_c) SET_C(cpu, 1);
    return a;
}

/* ============================================================
 * Block execution
 * ============================================================ */

int ir_exec_block(GB *gb, const IRBlock *ib)
{
    /* Allocate temporaries (stack-allocated for speed; cap at 256) */
    int nt = ib->n_tmps > 0 ? ib->n_tmps : 1;
    uint32_t *tmps = (uint32_t *)calloc((size_t)nt, sizeof(uint32_t));
    if (!tmps) return cpu_step(gb); /* allocation failure → fallback */

    int branch_taken = 0; /* tracks which path was taken for cycle accounting */

    for (int i = 0; i < ib->n_insns; i++) {
        const IRInsn *ins = &ib->insns[i];
        const IRVal *d  = &ins->dst;
        const IRVal *s0 = &ins->src[0];
        const IRVal *s1 = &ins->src[1];
        const IRVal *s2 = &ins->src[2];

        uint32_t v0 = get_val(s0, gb, tmps);
        uint32_t v1 = get_val(s1, gb, tmps);
        uint32_t v2 = get_val(s2, gb, tmps);

        switch (ins->op) {

        case IR_NOP: break;

        /* ── Data movement ── */
        case IR_MOV:
            set_val(d, gb, tmps, v0);
            break;

        case IR_LOAD8: {
            uint16_t addr = addr_val(s0, gb, tmps);
            uint8_t  byte = memory_read(gb, addr);
            set_val(d, gb, tmps, byte);
            break;
        }
        case IR_STORE8: {
            uint16_t addr = addr_val(s0, gb, tmps);
            memory_write(gb, addr, (uint8_t)v1);
            break;
        }

        /* ── 8-bit arithmetic ── */
        case IR_ADD8:  set_val(d, gb, tmps, (v0 + v1) & 0xFF);         break;
        case IR_ADC8:  set_val(d, gb, tmps, (v0 + v1 + v2) & 0xFF);    break;
        case IR_SUB8:  set_val(d, gb, tmps, (v0 - v1) & 0xFF);         break;
        case IR_SBC8:  set_val(d, gb, tmps, (v0 - v1 - v2) & 0xFF);    break;
        case IR_AND8:  set_val(d, gb, tmps, v0 & v1);                   break;
        case IR_OR8:   set_val(d, gb, tmps, v0 | v1);                   break;
        case IR_XOR8:  set_val(d, gb, tmps, v0 ^ v1);                   break;
        case IR_NOT8:  set_val(d, gb, tmps, (~v0) & 0xFF);              break;
        case IR_INC8:  set_val(d, gb, tmps, (v0 + 1) & 0xFF);          break;
        case IR_DEC8:  set_val(d, gb, tmps, (v0 - 1) & 0xFF);          break;

        /* ── 16-bit arithmetic ── */
        case IR_ADD16:    set_val(d, gb, tmps, (v0 + v1) & 0xFFFF);    break;
        case IR_INC16:    set_val(d, gb, tmps, (v0 + 1) & 0xFFFF);     break;
        case IR_DEC16:    set_val(d, gb, tmps, (v0 - 1) & 0xFFFF);     break;
        case IR_ADD16_SP: {
            /* dst = SP + sign_extend(src[0]) */
            int16_t offset = (int8_t)(uint8_t)v0;
            set_val(d, gb, tmps,
                    (uint32_t)((int32_t)gb->cpu.SP + offset) & 0xFFFF);
            break;
        }

        /* ── Bit / shift ── */
        case IR_SHL8:  set_val(d, gb, tmps, (v0 << 1) & 0xFF);         break;
        case IR_SHR8:  set_val(d, gb, tmps, v0 >> 1);                  break;
        case IR_SAR8:  set_val(d, gb, tmps, (uint32_t)((int8_t)(uint8_t)v0 >> 1) & 0xFF); break;
        case IR_RLC8:  set_val(d, gb, tmps, (((v0 << 1) | (v0 >> 7)) & 0xFF)); break;
        case IR_RRC8:  set_val(d, gb, tmps, (((v0 >> 1) | (v0 << 7)) & 0xFF)); break;
        case IR_RL8: {
            uint32_t cf = GET_C(&gb->cpu) ? 1u : 0u;
            set_val(d, gb, tmps, ((v0 << 1) | cf) & 0xFF);
            break;
        }
        case IR_RR8: {
            uint32_t cf = GET_C(&gb->cpu) ? 1u : 0u;
            set_val(d, gb, tmps, ((v0 >> 1) | (cf << 7)) & 0xFF);
            break;
        }
        case IR_SWAP8:
            set_val(d, gb, tmps, (((v0 & 0x0F) << 4) | ((v0 >> 4) & 0x0F)));
            break;
        case IR_BIT8:
            set_val(d, gb, tmps, (v0 >> (v1 & 7)) & 1);
            break;
        case IR_RES8:
            set_val(d, gb, tmps, v0 & ~(1u << (v1 & 7)));
            break;
        case IR_SET8:
            set_val(d, gb, tmps, v0 | (1u << (v1 & 7)));
            break;

        /* ── Flag computations ── */
        case IR_ZFLAG:
            set_val(d, gb, tmps, (v0 == 0) ? 1u : 0u);
            break;
        case IR_HFLAG_ADD:
            set_val(d, gb, tmps, ((v0 & 0xF) + (v1 & 0xF) + v2) > 0xF ? 1u : 0u);
            break;
        case IR_HFLAG_SUB:
            set_val(d, gb, tmps,
                    ((int)(v0 & 0xF) - (int)(v1 & 0xF) - (int)v2) < 0 ? 1u : 0u);
            break;
        case IR_CFLAG_ADD:
            set_val(d, gb, tmps, (v0 + v1 + v2) > 0xFF ? 1u : 0u);
            break;
        case IR_CFLAG_SUB:
            set_val(d, gb, tmps,
                    ((int)v0 - (int)v1 - (int)v2) < 0 ? 1u : 0u);
            break;
        case IR_HFLAG_A16:
            set_val(d, gb, tmps, ((v0 & 0xFFF) + (v1 & 0xFFF)) > 0xFFF ? 1u : 0u);
            break;
        case IR_CFLAG_A16:
            set_val(d, gb, tmps, ((uint32_t)v0 + v1) > 0xFFFF ? 1u : 0u);
            break;
        case IR_HFLAG_SP: {
            int8_t  e8   = (int8_t)(uint8_t)v0;
            uint32_t sp  = gb->cpu.SP;
            set_val(d, gb, tmps,
                    ((sp & 0xF) + (uint32_t)((uint8_t)e8 & 0xF)) > 0xF ? 1u : 0u);
            break;
        }
        case IR_CFLAG_SP: {
            int8_t  e8   = (int8_t)(uint8_t)v0;
            uint32_t sp  = gb->cpu.SP;
            set_val(d, gb, tmps,
                    ((sp & 0xFF) + (uint32_t)((uint8_t)e8)) > 0xFF ? 1u : 0u);
            break;
        }

        /* ── Stack ── */
        case IR_PUSH16: {
            uint16_t val16 = (uint16_t)v0;
            gb->cpu.SP -= 2;
            memory_write(gb, (uint16_t)(gb->cpu.SP + 1), (uint8_t)(val16 >> 8));
            memory_write(gb, gb->cpu.SP,                 (uint8_t)(val16 & 0xFF));
            break;
        }
        case IR_POP16: {
            uint8_t  lo  = memory_read(gb, gb->cpu.SP);
            uint8_t  hi  = memory_read(gb, (uint16_t)(gb->cpu.SP + 1));
            gb->cpu.SP  += 2;
            set_val(d, gb, tmps, (uint32_t)((hi << 8) | lo));
            break;
        }

        /* ── Control flow ── */
        case IR_JUMP:
            gb->cpu.PC = (uint16_t)v0;
            goto done;

        case IR_BRANCH: {
            /* src[0]=cond, src[1]=taken_target, src[2]=fall_target
             * Special cases:
             *   CALL cc: exit_type == EXIT_CALL_CC → push fall addr when taken
             *   RET  cc: taken_target == 0xFFFF    → pop PC when taken        */
            if (v0) {
                branch_taken = 1;
                if (s1->u16 == 0xFFFF) {
                    /* RET cc taken: pop PC */
                    uint8_t lo = memory_read(gb, gb->cpu.SP);
                    uint8_t hi = memory_read(gb, (uint16_t)(gb->cpu.SP + 1));
                    gb->cpu.SP += 2;
                    gb->cpu.PC = (uint16_t)((hi << 8) | lo);
                } else if (ib->exit_type == EXIT_CALL_CC) {
                    /* CALL cc taken: push fall-through addr, jump to target */
                    uint16_t ret_addr = (uint16_t)v2;
                    gb->cpu.SP -= 2;
                    memory_write(gb, (uint16_t)(gb->cpu.SP + 1),
                                 (uint8_t)(ret_addr >> 8));
                    memory_write(gb, gb->cpu.SP,
                                 (uint8_t)(ret_addr & 0xFF));
                    gb->cpu.PC = (uint16_t)v1;
                } else {
                    gb->cpu.PC = (uint16_t)v1;
                }
            } else {
                gb->cpu.PC = (uint16_t)v2;
            }
            goto done;
        }

        case IR_CALL: {
            /* src[0]=target, src[1]=return_addr */
            uint16_t ret_addr = (uint16_t)v1;
            gb->cpu.SP -= 2;
            memory_write(gb, (uint16_t)(gb->cpu.SP + 1),
                         (uint8_t)(ret_addr >> 8));
            memory_write(gb, gb->cpu.SP,
                         (uint8_t)(ret_addr & 0xFF));
            gb->cpu.PC = (uint16_t)v0;
            goto done;
        }

        case IR_RET: {
            uint8_t lo = memory_read(gb, gb->cpu.SP);
            uint8_t hi = memory_read(gb, (uint16_t)(gb->cpu.SP + 1));
            gb->cpu.SP += 2;
            gb->cpu.PC = (uint16_t)((hi << 8) | lo);
            goto done;
        }

        case IR_HALT:
            gb->cpu.halted = true;
            goto done;

        /* ── Misc ── */
        case IR_DAA:
            gb->cpu.A = exec_daa(&gb->cpu);
            break;

        case IR_IME_SET:
            gb->cpu.ime_pending = true;
            break;

        case IR_IME_CLR:
            gb->cpu.IME = false;
            gb->cpu.ime_pending = false;
            break;
        }
    }

done:
    free(tmps);
    /* Advance PC past the block if no explicit branch / jump was hit
     * (the loop exits naturally for fall-through blocks). */
    if (ib->exit_type == EXIT_FALLTHROUGH ||
        ib->exit_type == EXIT_CALL ||
        ib->exit_type == EXIT_RET ||
        ib->exit_type == EXIT_HALT) {
        /* PC was already set by JUMP/CALL/RET/HALT insn above, or for
         * fall-through the last IR insn left PC pointing at block entry.
         * For fall-through, advance to successor. */
        if (ib->exit_type == EXIT_FALLTHROUGH && ib->n_succs > 0)
            gb->cpu.PC = ib->succs[0];
    }

    gb->cpu.cycles += (uint64_t)(branch_taken ? ib->taken_cycles : ib->fall_cycles);
    return branch_taken ? ib->taken_cycles : ib->fall_cycles;
}

/* ============================================================
 * Block cache
 * ============================================================ */

void ir_cache_init(IRCache *c)
{
    memset(c, 0, sizeof(*c));
}

void ir_cache_free(IRCache *c)
{
    for (int i = 0; i < IR_CACHE_SLOTS; i++)
        if (c->slots[i].block) ir_free_block(c->slots[i].block);
    memset(c, 0, sizeof(*c));
}

/* Sorted index for binary-search lookup — built once on first miss. */
static uint32_t  cfg_sorted_keys[1 << 17]; /* (bank<<16)|entry, sorted */
static int       cfg_sorted_idx[1 << 17];  /* original cfg->blocks[] index */
static int       cfg_sorted_n   = 0;
static int       cfg_sorted_ok  = 0;

typedef struct { uint32_t key; int idx; } CfgKI;

static int cmp_cfgki(const void *a, const void *b) {
    const CfgKI *x = (const CfgKI *)a, *y = (const CfgKI *)b;
    return (x->key > y->key) - (x->key < y->key);
}

/* Build sorted index if not yet done.  Called at most once per process. */
static void cfg_build_index(const CFG *cfg)
{
    if (cfg_sorted_ok) return;
    int n = cfg->n_blocks;
    if (n > (int)(sizeof(cfg_sorted_keys)/sizeof(cfg_sorted_keys[0])))
        n = (int)(sizeof(cfg_sorted_keys)/sizeof(cfg_sorted_keys[0]));

    CfgKI *tmp = (CfgKI *)malloc((size_t)n * sizeof(CfgKI));
    if (!tmp) return; /* fallback: will use linear scan */
    for (int i = 0; i < n; i++) {
        tmp[i].key = ((uint32_t)cfg->blocks[i].bank << 16) | cfg->blocks[i].entry;
        tmp[i].idx = i;
    }
    qsort(tmp, (size_t)n, sizeof(CfgKI), cmp_cfgki);
    for (int i = 0; i < n; i++) {
        cfg_sorted_keys[i] = tmp[i].key;
        cfg_sorted_idx[i]  = tmp[i].idx;
    }
    cfg_sorted_n  = n;
    cfg_sorted_ok = 1;
    free(tmp);
}

static IRBlock *lift_block_from_cfg(const CFG *cfg, int bank, uint16_t addr)
{
    cfg_build_index(cfg);

    uint32_t target = ((uint32_t)bank << 16) | addr;

    /* Binary search */
    int lo = 0, hi = cfg_sorted_n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint32_t k = cfg_sorted_keys[mid];
        if (k == target)
            return ir_lift_block(&cfg->blocks[cfg_sorted_idx[mid]]);
        if (k < target) lo = mid + 1;
        else            hi = mid - 1;
    }
    return NULL;
}

const IRBlock *ir_cache_get(IRCache *c, const CFG *cfg, int bank, uint16_t addr)
{
    /* Use 0xFFFFFFFF as the sentinel key for slot "occupied but no block"
     * (address not in CFG).  Real key=0 maps to 0xFFFFFFFF too — handled
     * by the stored-key comparison below.                                  */
    uint32_t key      = ((uint32_t)bank << 16) | addr;
    uint32_t store_key = key ? key : (uint32_t)0xFFFFFFFF;
    uint32_t slot     = key & (IR_CACHE_SLOTS - 1);

    /* Linear probe */
    for (int probe = 0; probe < IR_CACHE_SLOTS; probe++) {
        IRCacheEntry *e = &c->slots[(slot + (uint32_t)probe) & (IR_CACHE_SLOTS - 1)];

        if (e->key == 0) {
            /* Empty slot — first time we see this address */
            IRBlock *ib = lift_block_from_cfg(cfg, bank, addr);
            /* Store result (even if NULL) so future lookups are O(1) */
            e->key   = store_key;
            e->block = ib; /* NULL = "not in CFG" sentinel */
            c->misses++;
            return ib;
        }
        if (e->key == store_key) {
            c->hits++;
            return e->block; /* may be NULL for non-CFG addresses */
        }
    }
    c->misses++;
    return NULL; /* table full — extremely unlikely */
}

/* ============================================================
 * High-level step
 * ============================================================ */

int ir_step(GB *gb, IRCache *cache, const CFG *cfg)
{
    CPU *cpu = &gb->cpu;

    /* EI delay: IME takes effect at the start of the step following EI */
    if (cpu->ime_pending) {
        cpu->IME = true;
        cpu->ime_pending = false;
    }

    /* Halted state or pending interrupt → fall back to cpu_step.
     * cpu_step handles HALT wakeup and interrupt service correctly. */
    uint8_t pending = gb->mem.io[0x0F] & gb->mem.IE & 0x1F;
    if (cpu->halted || (cpu->IME && pending)) {
        /* Undo the IME we just set so cpu_step re-does it properly.
         * (cpu_step also checks ime_pending at its start.) */
        if (cpu->IME && !cpu->halted) {
            /* We enabled IME here; set it back to pending so cpu_step
             * enables it again (no-op: it's already true, harmless). */
        }
        return cpu_step(gb);
    }

    /* Determine ROM bank for current PC */
    int      bank = (cpu->PC < 0x4000) ? 0 : (int)gb->mem.rom_bank;
    uint16_t pc   = cpu->PC;

    const IRBlock *ib = ir_cache_get(cache, cfg, bank, pc);
    if (!ib) return cpu_step(gb); /* fallback for dynamic/uncovered addresses */

    return ir_exec_block(gb, ib);
}
