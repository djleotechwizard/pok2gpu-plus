/*
 * GPU Batch Execution
 *
 * Each CUDA thread manages one complete GB state and runs the IR interpreter
 * loop, timer, PPU (no rendering), and OAM DMA entirely on the GPU.
 * States are sorted by current block key before dispatch to minimise
 * warp divergence when states follow different code paths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "batch.h"
#include "perf.cuh"
#include "../analysis/ir.h"
#include "../analysis/cfg.h"
/* hot_traces.cuh is included after gpu_mem_read/gpu_mem_write — see end of memory section */

/* CUDA_CHECK and NVTX macros are defined in perf.cuh */

/* ============================================================
 * Device helpers: flag / register accessors
 * ============================================================ */

#define GPU_FLAG_Z 0x80u
#define GPU_FLAG_N 0x40u
#define GPU_FLAG_H 0x20u
#define GPU_FLAG_C 0x10u

#define GPU_GET_Z(s)  (((s)->F & GPU_FLAG_Z) != 0)
#define GPU_GET_N(s)  (((s)->F & GPU_FLAG_N) != 0)
#define GPU_GET_H(s)  (((s)->F & GPU_FLAG_H) != 0)
#define GPU_GET_C(s)  (((s)->F & GPU_FLAG_C) != 0)

#define GPU_SET_Z(s,v) do { if(v) (s)->F |= GPU_FLAG_Z; else (s)->F &= ~GPU_FLAG_Z; } while(0)
#define GPU_SET_N(s,v) do { if(v) (s)->F |= GPU_FLAG_N; else (s)->F &= ~GPU_FLAG_N; } while(0)
#define GPU_SET_H(s,v) do { if(v) (s)->F |= GPU_FLAG_H; else (s)->F &= ~GPU_FLAG_H; } while(0)
#define GPU_SET_C(s,v) do { if(v) (s)->F |= GPU_FLAG_C; else (s)->F &= ~GPU_FLAG_C; } while(0)

__device__ __forceinline__ static uint16_t gpu_AF(const GBStateGPU *s)
    { return ((uint16_t)s->A << 8) | (s->F & 0xF0); }
__device__ __forceinline__ static uint16_t gpu_BC(const GBStateGPU *s)
    { return ((uint16_t)s->B << 8) | s->C; }
__device__ __forceinline__ static uint16_t gpu_DE(const GBStateGPU *s)
    { return ((uint16_t)s->D << 8) | s->E; }
__device__ __forceinline__ static uint16_t gpu_HL(const GBStateGPU *s)
    { return ((uint16_t)s->H << 8) | s->L; }

__device__ __forceinline__ static void gpu_set_AF(GBStateGPU *s, uint16_t v)
    { s->A = (uint8_t)(v >> 8); s->F = (uint8_t)(v & 0xF0); }
__device__ __forceinline__ static void gpu_set_BC(GBStateGPU *s, uint16_t v)
    { s->B = (uint8_t)(v >> 8); s->C = (uint8_t)(v & 0xFF); }
__device__ __forceinline__ static void gpu_set_DE(GBStateGPU *s, uint16_t v)
    { s->D = (uint8_t)(v >> 8); s->E = (uint8_t)(v & 0xFF); }
__device__ __forceinline__ static void gpu_set_HL(GBStateGPU *s, uint16_t v)
    { s->H = (uint8_t)(v >> 8); s->L = (uint8_t)(v & 0xFF); }

/* ============================================================
 * Device: Memory read/write
 * ============================================================ */

__device__ static uint8_t gpu_mem_read(
    GBStateGPU   *s,
    uint8_t      *vram,
    uint8_t      *wram,
    uint8_t      *ext_ram,     /* per-state slice, size = ext_ram_size */
    const uint8_t *rom,
    uint32_t      rom_size,
    uint32_t      ext_ram_size,
    uint16_t      addr)
{
    if (addr < 0x4000u) {
        return (addr < rom_size) ? rom[addr] : 0xFF;
    }
    if (addr < 0x8000u) {
        uint8_t bank = s->rom_bank ? s->rom_bank : 1;
        uint32_t off = (uint32_t)(bank & 0xFF) * 0x4000u;
        if (off >= rom_size) off = off % rom_size;
        return rom[off + (addr - 0x4000u)];
    }
    if (addr < 0xA000u) return vram[addr - 0x8000u];
    if (addr < 0xC000u) {
        if (!s->ram_enabled || !ext_ram || ext_ram_size == 0) return 0xFF;
        uint32_t off = (uint32_t)(s->ram_bank & 0x0F) * 0x2000u;
        if (off >= ext_ram_size) off = off % ext_ram_size;
        return ext_ram[off + (addr - 0xA000u)];
    }
    if (addr < 0xE000u) return wram[addr - 0xC000u];
    if (addr < 0xFE00u) return wram[(addr - 0xE000u) & 0x1FFF];
    if (addr < 0xFEA0u) return s->oam[addr - 0xFE00u];
    if (addr < 0xFF00u) return 0xFF;  /* unusable */
    if (addr < 0xFF80u) {
        uint8_t reg = (uint8_t)(addr - 0xFF00u);
        if (reg == 0x04) return (uint8_t)(s->timer_div >> 8);
        if (reg == 0x05) return s->timer_tima;
        if (reg == 0x06) return s->timer_tma;
        if (reg == 0x07) return s->timer_tac | 0xF8u;
        if (reg == 0x44) return (uint8_t)s->ppu_scanline;
        if (reg == 0x41) {
            uint8_t stat = s->io[0x41] & 0xF8u;
            stat |= (uint8_t)s->ppu_mode & 0x03u;
            if (s->ppu_scanline == (int32_t)s->io[0x45])
                stat |= 0x04u;
            return stat;
        }
        if (reg == 0x00) {
            /* JOYP: selection bits from io[0x00], button state from joypad_action.
             * Action byte: bit0=A bit1=B bit2=sel bit3=start
             *              bit4=right bit5=left bit6=up bit7=down (1=pressed)
             * JOYP bits 0-3 are active-LOW: 0 = pressed. */
            uint8_t sel = s->io[0x00];
            uint8_t lo  = 0x0Fu; /* default: nothing pressed */
            if (!(sel & 0x20u))  /* buttons selected */
                lo &= ~(s->joypad_action & 0x0Fu);        /* A/B/sel/start */
            if (!(sel & 0x10u))  /* d-pad selected */
                lo &= ~((s->joypad_action >> 4) & 0x0Fu); /* right/left/up/down */
            return (sel & 0x30u) | 0xC0u | lo;
        }
        return s->io[reg];
    }
    if (addr < 0xFFFFu) return s->hram[addr - 0xFF80u];
    return s->IE;
}

__device__ static void gpu_mem_write(
    GBStateGPU    *s,
    uint8_t       *vram,
    uint8_t       *wram,
    uint8_t       *ext_ram,
    const uint8_t *rom,
    uint32_t       rom_size,
    uint32_t       ext_ram_size,
    uint16_t       addr,
    uint8_t        val)
{
    /* MBC register writes (0x0000–0x7FFF) */
    if (addr < 0x2000u) {
        s->ram_enabled = ((val & 0x0Fu) == 0x0Au) ? 1 : 0;
        return;
    }
    if (addr < 0x4000u) {
        if (s->mbc_type == 1) {           /* MBC1 */
            uint8_t b = val & 0x1F; if (!b) b = 1;
            s->rom_bank = (s->rom_bank & 0x60u) | b;
        } else if (s->mbc_type == 3) {    /* MBC3 */
            uint8_t b = val & 0x7Fu; if (!b) b = 1;
            s->rom_bank = b;
        } else if (s->mbc_type == 5) {    /* MBC5 */
            s->rom_bank = val;
        }
        return;
    }
    if (addr < 0x6000u) {
        if (s->mbc_type == 1) {
            if (s->mbc1_mode == 0)
                s->rom_bank = (s->rom_bank & 0x1Fu) | (uint8_t)((val & 0x03u) << 5);
            else
                s->ram_bank = val & 0x03u;
        } else if (s->mbc_type == 3) {
            if (val <= 0x03u) s->ram_bank = val;
        } else if (s->mbc_type == 5) {
            s->ram_bank = val & 0x0Fu;
        }
        return;
    }
    if (addr < 0x8000u) {
        if (s->mbc_type == 1) s->mbc1_mode = val & 0x01;
        return;
    }
    if (addr < 0xA000u) { vram[addr - 0x8000u] = val; return; }
    if (addr < 0xC000u) {
        if (s->ram_enabled && ext_ram && ext_ram_size > 0) {
            uint32_t off = (uint32_t)(s->ram_bank & 0x0F) * 0x2000u;
            if (off >= ext_ram_size) off = off % ext_ram_size;
            ext_ram[off + (addr - 0xA000u)] = val;
        }
        return;
    }
    if (addr < 0xE000u) { wram[addr - 0xC000u] = val; return; }
    if (addr < 0xFE00u) { wram[(addr - 0xE000u) & 0x1FFF] = val; return; }
    if (addr < 0xFEA0u) { s->oam[addr - 0xFE00u] = val; return; }
    if (addr < 0xFF00u) return;
    if (addr < 0xFF80u) {
        uint8_t reg = (uint8_t)(addr - 0xFF00u);
        switch (reg) {
            case 0x04: /* DIV */ s->timer_div = 0; s->io[0x04] = 0; return;
            case 0x05: /* TIMA */ s->timer_tima = val; return;
            case 0x06: /* TMA  */ s->timer_tma  = val; return;
            case 0x07: /* TAC  */ s->timer_tac  = val & 0x07u; return;
            case 0x0F: /* IF   */ s->io[0x0F] = val | 0xE0u; return;
            case 0x44: /* LY   */ return; /* read-only */
            case 0x41: /* STAT */
                s->io[0x41] = (s->io[0x41] & 0x07u) | (val & 0x78u);
                return;
            case 0x46: { /* DMA — bulk-copy all 160 OAM bytes immediately */
                uint16_t src_base = (uint16_t)((uint16_t)val << 8);
                s->io[0x46] = val;
                for (int _di = 0; _di < 160; _di++)
                    s->oam[_di] = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                               src_base + (uint16_t)_di);
                s->dma_active = 1;
                s->dma_src    = src_base;
                /* dma_index repurposed as T-cycles remaining (160 M-cycles = 640 T-cycles) */
                s->dma_index  = 640;
                return;
            }
            default: s->io[reg] = val; return;
        }
    }
    if (addr < 0xFFFFu) { s->hram[addr - 0xFF80u] = val; return; }
    s->IE = val;
}

/* JIT hot-trace dispatch — included here so gpu_mem_read/write are in scope */
#include "hot_traces.cuh"

/* ============================================================
 * Device: Interrupt service
 * ============================================================ */

__device__ static int gpu_service_interrupt(
    GBStateGPU   *s,
    uint8_t      *vram,
    uint8_t      *wram,
    uint8_t      *ext_ram,
    const uint8_t *rom,
    uint32_t      rom_size,
    uint32_t      ext_ram_size)
{
    uint8_t pending = s->io[0x0F] & s->IE & 0x1F;
    for (int i = 0; i < 5; i++) {
        if (pending & (1 << i)) {
            s->io[0x0F] &= (uint8_t)(~(1 << i));
            s->IME = 0;
            /* push PC */
            s->SP -= 2;
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          (uint16_t)(s->SP + 1), (uint8_t)(s->PC >> 8));
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          s->SP,                 (uint8_t)(s->PC & 0xFF));
            s->PC = (uint16_t)(0x40 + i * 8);
            return 20;
        }
    }
    return 0;
}

/* ============================================================
 * Device: Timer step
 * ============================================================ */

/* ── Timer step: O(1) analytical version ─────────────────────
 * Old implementation iterated `cycles` times (up to 16+ per block).
 * New version uses the falling-edge count formula:
 *   edges = floor((div+N) / period) - floor(div / period)
 * where period = 2^(bit+1) for the TAC-selected bit.
 * This reduces a hot O(cycles) loop to O(1) arithmetic.        */
__device__ static void gpu_timer_step(GBStateGPU *s, int cycles)
{
    /* ── Pending overflow reload (4-cycle delay after TIMA wraps) ── */
    if (s->timer_overflow) {
        s->timer_overflow_delay -= cycles;
        if (s->timer_overflow_delay <= 0) {
            s->timer_overflow = 0;
            s->timer_tima     = s->timer_tma;
            s->io[0x0F]      |= 0x04u; /* INT_TIMER */
        }
        /* timer_div still advances; skip edge counting this call */
        s->timer_div += (uint16_t)cycles;
        s->io[0x04] = (uint8_t)(s->timer_div >> 8);
        return;
    }

    /* ── Count falling edges analytically ── */
    if (s->timer_tac & 0x04u) {            /* timer enabled */
        const int tac_bit[4] = { 9, 3, 5, 7 };
        int      bit    = tac_bit[s->timer_tac & 0x03u];
        uint32_t period = 1u << (bit + 1);
        uint32_t div    = (uint32_t)s->timer_div;

        uint32_t falling = (div + (uint32_t)cycles) / period
                         -  div / period;
        if (falling > 0u) {
            uint32_t new_tima = (uint32_t)s->timer_tima + falling;
            if (new_tima > 0xFFu) {
                /* TIMA wrapped — set overflow with 4-cycle reload delay */
                s->timer_tima           = (uint8_t)(new_tima & 0xFFu);
                s->timer_overflow       = 1;
                s->timer_overflow_delay = 4;
            } else {
                s->timer_tima = (uint8_t)new_tima;
            }
        }
    }

    s->timer_div += (uint16_t)cycles;
    s->io[0x04] = (uint8_t)(s->timer_div >> 8);
}

/* ============================================================
 * Device: PPU step (no rendering, just timing + interrupts)
 * ============================================================ */

#define GPU_DOTS_PER_LINE  456
#define GPU_MODE2_END       80
#define GPU_MODE3_END      252
#define GPU_VBLANK_START   144
#define GPU_TOTAL_LINES    154

__device__ static void gpu_check_stat_irq(GBStateGPU *s)
{
    uint8_t stat = s->io[0x41];
    uint8_t lcdc = s->io[0x40];
    if (!(lcdc & 0x80u)) return;

    bool fire = false;
    switch (s->ppu_mode) {
        case 0: fire = (stat & 0x08u) != 0; break; /* HBLANK */
        case 1: fire = (stat & 0x10u) != 0; break; /* VBLANK */
        case 2: fire = (stat & 0x20u) != 0; break; /* OAM    */
        default: break;
    }
    if ((stat & 0x40u) && (s->ppu_scanline == (int32_t)s->io[0x45]))
        fire = true;
    if (fire) s->io[0x0F] |= 0x02u; /* INT_STAT */
}

/* ============================================================
 * Device: HALT fast-forward helper
 * Compute T-cycles until the next interrupt event that IE allows.
 * Used to skip the CPU-halted spinning in big jumps rather than
 * stepping 4 cycles at a time (which costs ~17,500 calls/frame).
 * ============================================================ */

__device__ __forceinline__ static int gpu_cycles_to_wake(const GBStateGPU *s)
{
    uint8_t ie = s->IE;
    int min_cyc = GPU_TOTAL_LINES * GPU_DOTS_PER_LINE; /* 70224: full frame cap */

    /* ── VBlank (IE bit 0) ── */
    if (ie & 0x01u) {
        int cur = s->ppu_scanline * GPU_DOTS_PER_LINE + s->ppu_dot;
        int vb  = GPU_VBLANK_START * GPU_DOTS_PER_LINE;
        int d   = (cur < vb) ? (vb - cur)
                             : (GPU_TOTAL_LINES * GPU_DOTS_PER_LINE - cur + vb);
        if (d < min_cyc) min_cyc = d;
    }

    /* ── STAT (IE bit 1) — conservative: next scanline boundary ── */
    if (ie & 0x02u) {
        int rem = GPU_DOTS_PER_LINE - (s->ppu_dot % GPU_DOTS_PER_LINE);
        if (rem < min_cyc) min_cyc = rem;
    }

    /* ── Timer (IE bit 2) ── */
    if ((ie & 0x04u) && (s->timer_tac & 0x04u)) {
        const int tac_bit[4] = { 9, 3, 5, 7 };
        int      bit    = tac_bit[s->timer_tac & 0x03u];
        uint32_t period = 1u << (bit + 1);
        uint32_t div    = (uint32_t)s->timer_div;
        /* T-cycles until the next falling edge of the selected bit */
        uint32_t next_edge = period - (div & (period - 1u));
        /* Additional edges needed to overflow TIMA */
        uint32_t edges_left = 0x100u - (uint32_t)s->timer_tima;
        uint32_t timer_cyc  = next_edge + (edges_left > 1u ? (edges_left - 1u) * period : 0u);
        if ((int)timer_cyc < min_cyc) min_cyc = (int)timer_cyc;
    }

    /* Clamp: always advance at least 4 T-cycles, align to 4 */
    if (min_cyc < 4) min_cyc = 4;
    return min_cyc & ~3;
}

__device__ static void gpu_ppu_step(GBStateGPU *s, int cycles)
{
    uint8_t lcdc = s->io[0x40];
    if (!(lcdc & 0x80u)) {
        /* LCD disabled: accumulate dots and count frames so ppu_frame advances
         * at the same rate as the CPU emulator (prevents GPU running extra game
         * cycles during map transitions when the game briefly disables LCD). */
        s->ppu_dot += cycles;
        if (s->ppu_dot >= GPU_DOTS_PER_LINE * GPU_TOTAL_LINES) {
            s->ppu_dot -= GPU_DOTS_PER_LINE * GPU_TOTAL_LINES;
            s->ppu_frame++;
        }
        s->ppu_scanline = 0;
        s->io[0x44]     = 0;
        s->io[0x41]     = (s->io[0x41] & 0xF8u) | 0; /* HBLANK */
        s->ppu_mode     = 0;
        return;
    }

    s->ppu_dot += cycles;

    while (s->ppu_dot >= GPU_DOTS_PER_LINE) {
        s->ppu_dot -= GPU_DOTS_PER_LINE;
        s->ppu_scanline++;

        if (s->ppu_scanline == GPU_VBLANK_START) {
            s->ppu_mode = 1; /* VBLANK */
            s->io[0x41] = (s->io[0x41] & 0xF8u) | 1;
            s->io[0x0F] |= 0x01u; /* INT_VBLANK */
            gpu_check_stat_irq(s);
            s->ppu_frame++;
        } else if (s->ppu_scanline >= GPU_TOTAL_LINES) {
            s->ppu_scanline = 0;
            s->ppu_mode = 2; /* OAM */
            s->io[0x41] = (s->io[0x41] & 0xF8u) | 2;
            gpu_check_stat_irq(s);
        }

        s->io[0x44] = (uint8_t)s->ppu_scanline;

        bool coinc = (s->ppu_scanline == (int32_t)s->io[0x45]);
        if (coinc) s->io[0x41] |=  0x04u;
        else       s->io[0x41] &= ~0x04u;
        if (coinc && (s->io[0x41] & 0x40u))
            s->io[0x0F] |= 0x02u; /* INT_STAT */
    }

    if (s->ppu_scanline < GPU_VBLANK_START) {
        int new_mode;
        if      (s->ppu_dot < GPU_MODE2_END) new_mode = 2; /* OAM  */
        else if (s->ppu_dot < GPU_MODE3_END) new_mode = 3; /* DRAW */
        else                                 new_mode = 0; /* HBLANK */

        if (new_mode != s->ppu_mode) {
            s->ppu_mode = new_mode;
            s->io[0x41] = (s->io[0x41] & 0xF8u) | (uint8_t)new_mode;
            gpu_check_stat_irq(s);
        }
    }
}

/* ============================================================
 * Device: OAM DMA tick
 * ============================================================ */

/* OAM DMA tick — OAM bytes were bulk-copied at DMA-start; just count down
 * the bus-busy duration (640 T-cycles = 160 M-cycles).
 * Signature unchanged so all call sites compile without modification. */
__device__ static void gpu_dma_tick(
    GBStateGPU   *s,
    uint8_t      * /*ext_ram*/,
    const uint8_t * /*rom*/,
    uint32_t       /*rom_size*/,
    uint32_t       /*ext_ram_size*/,
    int            cycles)
{
    if (!s->dma_active) return;
    s->dma_index -= cycles;
    if (s->dma_index <= 0) s->dma_active = 0;
}

/* Combined per-block tick called from JIT traces.
 * __noinline__ prevents NVCC from expanding the three while-loop functions
 * hundreds of times across all trace functions, which would crash the compiler. */
__device__ __noinline__ static void gpu_trace_subsys_tick(
    GBStateGPU *s, uint8_t *ext_ram, const uint8_t *rom,
    uint32_t rom_size, uint32_t ext_ram_size, int cyc)
{
    gpu_ppu_step(s, cyc);
    gpu_timer_step(s, cyc);
    gpu_dma_tick(s, ext_ram, rom, rom_size, ext_ram_size, cyc);
}

/* ============================================================
 * Device: IR value get/set
 * ============================================================ */

#define GPU_MAX_TMPS 128u

__device__ static uint32_t gpu_get_val(
    const IRVal *v, const GBStateGPU *s, const uint32_t *tmps)
{
    switch (v->kind) {
        case IRV_NONE:  return 0;
        case IRV_A:     return s->A;
        case IRV_B:     return s->B;
        case IRV_C:     return s->C;
        case IRV_D:     return s->D;
        case IRV_E:     return s->E;
        case IRV_H:     return s->H;
        case IRV_L:     return s->L;
        case IRV_AF:    return gpu_AF(s);
        case IRV_BC:    return gpu_BC(s);
        case IRV_DE:    return gpu_DE(s);
        case IRV_HL:    return gpu_HL(s);
        case IRV_SP:    return s->SP;
        case IRV_ZF:    return GPU_GET_Z(s) ? 1u : 0u;
        case IRV_NF:    return GPU_GET_N(s) ? 1u : 0u;
        case IRV_HF:    return GPU_GET_H(s) ? 1u : 0u;
        case IRV_CF:    return GPU_GET_C(s) ? 1u : 0u;
        case IRV_IMM8:  return v->u8;
        case IRV_IMM16: return v->u16;
        case IRV_TMP:   return (v->idx >= 0 && (uint32_t)v->idx < GPU_MAX_TMPS)
                                ? tmps[v->idx] : 0u;
        default:        return 0;
    }
}

__device__ static void gpu_set_val(
    const IRVal *v, GBStateGPU *s, uint32_t *tmps, uint32_t val)
{
    switch (v->kind) {
        case IRV_A:  s->A = (uint8_t)val;  break;
        case IRV_B:  s->B = (uint8_t)val;  break;
        case IRV_C:  s->C = (uint8_t)val;  break;
        case IRV_D:  s->D = (uint8_t)val;  break;
        case IRV_E:  s->E = (uint8_t)val;  break;
        case IRV_H:  s->H = (uint8_t)val;  break;
        case IRV_L:  s->L = (uint8_t)val;  break;
        case IRV_AF: gpu_set_AF(s, (uint16_t)val); break;
        case IRV_BC: gpu_set_BC(s, (uint16_t)val); break;
        case IRV_DE: gpu_set_DE(s, (uint16_t)val); break;
        case IRV_HL: gpu_set_HL(s, (uint16_t)val); break;
        case IRV_SP: s->SP = (uint16_t)val; break;
        case IRV_ZF: GPU_SET_Z(s, val); break;
        case IRV_NF: GPU_SET_N(s, val); break;
        case IRV_HF: GPU_SET_H(s, val); break;
        case IRV_CF: GPU_SET_C(s, val); break;
        case IRV_TMP:
            if (v->idx >= 0 && (uint32_t)v->idx < GPU_MAX_TMPS)
                tmps[v->idx] = val;
            break;
        default: break;
    }
}

/* Address resolution for LOAD8/STORE8 */
__device__ static uint16_t gpu_addr_val(
    const IRVal *v, const GBStateGPU *s, const uint32_t *tmps)
{
    switch (v->kind) {
        case IRV_A:  return (uint16_t)(0xFF00u | s->A);
        case IRV_B:  return (uint16_t)(0xFF00u | s->B);
        case IRV_C:  return (uint16_t)(0xFF00u | s->C);
        case IRV_D:  return (uint16_t)(0xFF00u | s->D);
        case IRV_E:  return (uint16_t)(0xFF00u | s->E);
        case IRV_H:  return (uint16_t)(0xFF00u | s->H);
        case IRV_L:  return (uint16_t)(0xFF00u | s->L);
        case IRV_BC:    return gpu_BC(s);
        case IRV_DE:    return gpu_DE(s);
        case IRV_HL:    return gpu_HL(s);
        case IRV_SP:    return s->SP;
        case IRV_IMM16: return v->u16;
        case IRV_IMM8:  return (uint16_t)(0xFF00u | v->u8);
        case IRV_TMP: {
            uint32_t x = (v->idx >= 0 && (uint32_t)v->idx < GPU_MAX_TMPS)
                         ? tmps[v->idx] : 0u;
            return (x < 0x100u) ? (uint16_t)(0xFF00u | x) : (uint16_t)x;
        }
        default: return 0;
    }
}

/* DAA correction */
__device__ static uint8_t gpu_exec_daa(GBStateGPU *s)
{
    uint8_t a = s->A;
    uint8_t corr = 0;
    int set_c = 0;

    if (!GPU_GET_N(s)) {
        if (GPU_GET_H(s) || (a & 0x0F) > 9)  corr |= 0x06;
        if (GPU_GET_C(s) || a > 0x99)         { corr |= 0x60; set_c = 1; }
        a = (uint8_t)(a + corr);
    } else {
        if (GPU_GET_H(s)) corr |= 0x06;
        if (GPU_GET_C(s)) { corr |= 0x60; set_c = 1; }
        a = (uint8_t)(a - corr);
    }
    GPU_SET_Z(s, a == 0);
    GPU_SET_H(s, 0);
    if (set_c) GPU_SET_C(s, 1);
    return a;
}

/* ============================================================
 * Device: IR block execution
 * ============================================================ */

/* ── Templatized block executor ───────────────────────────────
 * NTMPS is the compile-time stack allocation for temporaries.
 * Specialising on {8,16,32,64,128} lets the compiler eliminate
 * dead stack space and avoid local-memory spills for the majority
 * of blocks (Pokemon Red blocks usually need ≤ 16 temps).
 * The bounds check in gpu_get_val / gpu_set_val uses GPU_MAX_TMPS
 * as a safety net; IR correctness guarantees idx < blk->n_tmps
 * ≤ NTMPS, so no out-of-bounds access occurs in practice.      */
template<int NTMPS>
__device__ static int gpu_exec_block_T(
    GBStateGPU    *s,
    uint8_t       *vram,
    uint8_t       *wram,
    uint8_t       *ext_ram,
    const uint8_t *rom,
    uint32_t       rom_size,
    uint32_t       ext_ram_size,
    const IRBlockInfoGPU *blk,
    const IRInsn  *all_insns)
{
    uint32_t tmps[NTMPS];
    for (int i = 0; i < NTMPS; i++) tmps[i] = 0;

    int branch_taken = 0;
    const IRInsn *insns = all_insns + blk->insn_offset;

    for (int i = 0; i < blk->n_insns; i++) {
        const IRInsn *ins = &insns[i];
        const IRVal *d  = &ins->dst;
        const IRVal *s0 = &ins->src[0];
        const IRVal *s1 = &ins->src[1];
        const IRVal *s2 = &ins->src[2];

        uint32_t v0 = gpu_get_val(s0, s, tmps);
        uint32_t v1 = gpu_get_val(s1, s, tmps);
        uint32_t v2 = gpu_get_val(s2, s, tmps);

        switch (ins->op) {

        case IR_NOP: break;

        /* ── Data movement ── */
        case IR_MOV:
            gpu_set_val(d, s, tmps, v0);
            break;
        case IR_LOAD8: {
            uint16_t addr = gpu_addr_val(s0, s, tmps);
            uint8_t  byte = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, addr);
            gpu_set_val(d, s, tmps, byte);
            break;
        }
        case IR_STORE8: {
            uint16_t addr = gpu_addr_val(s0, s, tmps);
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, addr, (uint8_t)v1);
            break;
        }

        /* ── 8-bit arithmetic ── */
        case IR_ADD8:  gpu_set_val(d, s, tmps, (v0 + v1) & 0xFF);          break;
        case IR_ADC8:  gpu_set_val(d, s, tmps, (v0 + v1 + v2) & 0xFF);     break;
        case IR_SUB8:  gpu_set_val(d, s, tmps, (v0 - v1) & 0xFF);          break;
        case IR_SBC8:  gpu_set_val(d, s, tmps, (v0 - v1 - v2) & 0xFF);     break;
        case IR_AND8:  gpu_set_val(d, s, tmps, v0 & v1);                    break;
        case IR_OR8:   gpu_set_val(d, s, tmps, v0 | v1);                    break;
        case IR_XOR8:  gpu_set_val(d, s, tmps, v0 ^ v1);                    break;
        case IR_NOT8:  gpu_set_val(d, s, tmps, (~v0) & 0xFF);               break;
        case IR_INC8:  gpu_set_val(d, s, tmps, (v0 + 1) & 0xFF);           break;
        case IR_DEC8:  gpu_set_val(d, s, tmps, (v0 - 1) & 0xFF);           break;

        /* ── 16-bit arithmetic ── */
        case IR_ADD16:    gpu_set_val(d, s, tmps, (v0 + v1) & 0xFFFF);     break;
        case IR_INC16:    gpu_set_val(d, s, tmps, (v0 + 1) & 0xFFFF);      break;
        case IR_DEC16:    gpu_set_val(d, s, tmps, (v0 - 1) & 0xFFFF);      break;
        case IR_ADD16_SP: {
            int16_t offset = (int8_t)(uint8_t)v0;
            gpu_set_val(d, s, tmps,
                        (uint32_t)((int32_t)(int16_t)s->SP + offset) & 0xFFFF);
            break;
        }

        /* ── Bit / shift ── */
        case IR_SHL8:  gpu_set_val(d, s, tmps, (v0 << 1) & 0xFF);          break;
        case IR_SHR8:  gpu_set_val(d, s, tmps, v0 >> 1);                   break;
        case IR_SAR8:  gpu_set_val(d, s, tmps,
                           (uint32_t)((int8_t)(uint8_t)v0 >> 1) & 0xFF);   break;
        case IR_RLC8:  gpu_set_val(d, s, tmps,
                           (((v0 << 1) | (v0 >> 7)) & 0xFF));              break;
        case IR_RRC8:  gpu_set_val(d, s, tmps,
                           (((v0 >> 1) | (v0 << 7)) & 0xFF));              break;
        case IR_RL8: {
            uint32_t cf = GPU_GET_C(s) ? 1u : 0u;
            gpu_set_val(d, s, tmps, ((v0 << 1) | cf) & 0xFF);
            break;
        }
        case IR_RR8: {
            uint32_t cf = GPU_GET_C(s) ? 1u : 0u;
            gpu_set_val(d, s, tmps, ((v0 >> 1) | (cf << 7)) & 0xFF);
            break;
        }
        case IR_SWAP8:
            gpu_set_val(d, s, tmps,
                (((v0 & 0x0F) << 4) | ((v0 >> 4) & 0x0F)));
            break;
        case IR_BIT8:
            gpu_set_val(d, s, tmps, (v0 >> (v1 & 7)) & 1);
            break;
        case IR_RES8:
            gpu_set_val(d, s, tmps, v0 & ~(1u << (v1 & 7)));
            break;
        case IR_SET8:
            gpu_set_val(d, s, tmps, v0 | (1u << (v1 & 7)));
            break;

        /* ── Flag computations ── */
        case IR_ZFLAG:
            gpu_set_val(d, s, tmps, (v0 == 0) ? 1u : 0u);
            break;
        case IR_HFLAG_ADD:
            gpu_set_val(d, s, tmps,
                ((v0 & 0xF) + (v1 & 0xF) + v2) > 0xF ? 1u : 0u);
            break;
        case IR_HFLAG_SUB:
            gpu_set_val(d, s, tmps,
                ((int)(v0 & 0xF) - (int)(v1 & 0xF) - (int)v2) < 0
                ? 1u : 0u);
            break;
        case IR_CFLAG_ADD:
            gpu_set_val(d, s, tmps, (v0 + v1 + v2) > 0xFF ? 1u : 0u);
            break;
        case IR_CFLAG_SUB:
            gpu_set_val(d, s, tmps,
                ((int)v0 - (int)v1 - (int)v2) < 0 ? 1u : 0u);
            break;
        case IR_HFLAG_A16:
            gpu_set_val(d, s, tmps,
                ((v0 & 0xFFF) + (v1 & 0xFFF)) > 0xFFF ? 1u : 0u);
            break;
        case IR_CFLAG_A16:
            gpu_set_val(d, s, tmps,
                ((uint32_t)v0 + v1) > 0xFFFF ? 1u : 0u);
            break;
        case IR_HFLAG_SP: {
            int8_t  e8 = (int8_t)(uint8_t)v0;
            uint32_t sp = s->SP;
            gpu_set_val(d, s, tmps,
                ((sp & 0xF) + (uint32_t)((uint8_t)e8 & 0xF)) > 0xF ? 1u : 0u);
            break;
        }
        case IR_CFLAG_SP: {
            int8_t  e8 = (int8_t)(uint8_t)v0;
            uint32_t sp = s->SP;
            gpu_set_val(d, s, tmps,
                ((sp & 0xFF) + (uint32_t)(uint8_t)e8) > 0xFF ? 1u : 0u);
            break;
        }

        /* ── Stack ── */
        case IR_PUSH16: {
            uint16_t val16 = (uint16_t)v0;
            s->SP -= 2;
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          (uint16_t)(s->SP + 1), (uint8_t)(val16 >> 8));
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          s->SP, (uint8_t)(val16 & 0xFF));
            break;
        }
        case IR_POP16: {
            uint8_t lo = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, s->SP);
            uint8_t hi = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                      (uint16_t)(s->SP + 1));
            s->SP += 2;
            gpu_set_val(d, s, tmps, (uint32_t)((hi << 8) | lo));
            break;
        }

        /* ── Control flow ── */
        case IR_JUMP:
            s->PC = (uint16_t)v0;
            goto done;

        case IR_BRANCH: {
            if (v0) {
                branch_taken = 1;
                if (s1->u16 == 0xFFFF) {
                    /* RET cc taken: pop PC */
                    uint8_t lo = gpu_mem_read(s, vram, wram, ext_ram, rom,
                                              rom_size, ext_ram_size, s->SP);
                    uint8_t hi = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                              (uint16_t)(s->SP + 1));
                    s->SP += 2;
                    s->PC = (uint16_t)((hi << 8) | lo);
                } else if (blk->exit_type == (int)EXIT_CALL_CC) {
                    /* CALL cc taken: push fall-through addr, jump to target */
                    uint16_t ret_addr = (uint16_t)v2;
                    s->SP -= 2;
                    gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                  (uint16_t)(s->SP + 1), (uint8_t)(ret_addr >> 8));
                    gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                  s->SP, (uint8_t)(ret_addr & 0xFF));
                    s->PC = (uint16_t)v1;
                } else {
                    s->PC = (uint16_t)v1;
                }
            } else {
                s->PC = (uint16_t)v2;
            }
            goto done;
        }

        case IR_CALL: {
            uint16_t ret_addr = (uint16_t)v1;
            s->SP -= 2;
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          (uint16_t)(s->SP + 1), (uint8_t)(ret_addr >> 8));
            gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                          s->SP, (uint8_t)(ret_addr & 0xFF));
            s->PC = (uint16_t)v0;
            goto done;
        }

        case IR_RET: {
            uint8_t lo = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, s->SP);
            uint8_t hi = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                      (uint16_t)(s->SP + 1));
            s->SP += 2;
            s->PC = (uint16_t)((hi << 8) | lo);
            goto done;
        }

        case IR_HALT:
            s->halted = 1;
            goto done;

        /* ── Misc ── */
        case IR_DAA:
            s->A = gpu_exec_daa(s);
            break;
        case IR_IME_SET:
            s->ime_pending = 1;
            break;
        case IR_IME_CLR:
            s->IME = 0;
            s->ime_pending = 0;
            break;

        default: break;
        } /* switch */
    } /* for */

done:
    /* Advance PC for fall-through blocks */
    if (blk->exit_type == (int)EXIT_FALLTHROUGH && blk->succs[0] != 0)
        s->PC = blk->succs[0];

    int cyc = branch_taken ? blk->taken_cycles : blk->fall_cycles;
    s->cycles += (uint64_t)cyc;
    return cyc;
}

/* Dispatcher — selects the right template instantiation so the
 * compiler allocates only as many stack slots as the block needs. */
__device__ __noinline__ static int gpu_exec_block(
    GBStateGPU    *s,
    uint8_t       *vram,
    uint8_t       *wram,
    uint8_t       *ext_ram,
    const uint8_t *rom,
    uint32_t       rom_size,
    uint32_t       ext_ram_size,
    const IRBlockInfoGPU *blk,
    const IRInsn  *all_insns)
{
    int nt = blk->n_tmps;
    if (nt <=  8) return gpu_exec_block_T< 8>(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,blk,all_insns);
    if (nt <= 16) return gpu_exec_block_T<16>(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,blk,all_insns);
    if (nt <= 32) return gpu_exec_block_T<32>(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,blk,all_insns);
    if (nt <= 64) return gpu_exec_block_T<64>(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,blk,all_insns);
    return gpu_exec_block_T<128>(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,blk,all_insns);
}

/* ============================================================
 * Device: Full SM83 fallback decoder (for non-CFG addresses)
 * Matches cpu_step semantics exactly for all 256+256 opcodes.
 * ============================================================ */

/* Register r (0=B 1=C 2=D 3=E 4=H 5=L 6=(HL) 7=A) */
__device__ static uint8_t gpu_reg_r(GBStateGPU *s, uint8_t *vram, uint8_t *wram, uint8_t *ext_ram,
    const uint8_t *rom, uint32_t rom_size, uint32_t ext_ram_size, int r)
{
    switch (r & 7) {
        case 0: return s->B;
        case 1: return s->C;
        case 2: return s->D;
        case 3: return s->E;
        case 4: return s->H;
        case 5: return s->L;
        case 6: return gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, gpu_HL(s));
        case 7: return s->A;
    }
    return 0;
}
__device__ static void gpu_reg_w(GBStateGPU *s, uint8_t *vram, uint8_t *wram, uint8_t *ext_ram,
    const uint8_t *rom, uint32_t rom_size, uint32_t ext_ram_size, int r, uint8_t v)
{
    switch (r & 7) {
        case 0: s->B = v; return;
        case 1: s->C = v; return;
        case 2: s->D = v; return;
        case 3: s->E = v; return;
        case 4: s->H = v; return;
        case 5: s->L = v; return;
        case 6: gpu_mem_write(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, gpu_HL(s), v); return;
        case 7: s->A = v; return;
    }
}

/* 16-bit register pair rr (0=BC 1=DE 2=HL 3=SP) */
__device__ static uint16_t gpu_rp(const GBStateGPU *s, int rp)
{
    switch (rp & 3) {
        case 0: return gpu_BC(s);
        case 1: return gpu_DE(s);
        case 2: return gpu_HL(s);
        case 3: return s->SP;
    }
    return 0;
}
__device__ static void gpu_rp_w(GBStateGPU *s, int rp, uint16_t v)
{
    switch (rp & 3) {
        case 0: gpu_set_BC(s, v); return;
        case 1: gpu_set_DE(s, v); return;
        case 2: gpu_set_HL(s, v); return;
        case 3: s->SP = v; return;
    }
}
/* rp2 for PUSH/POP: 0=BC 1=DE 2=HL 3=AF */
__device__ static uint16_t gpu_rp2(const GBStateGPU *s, int rp)
{
    if ((rp & 3) == 3) return gpu_AF(s);
    return gpu_rp(s, rp);
}
__device__ static void gpu_rp2_w(GBStateGPU *s, int rp, uint16_t v)
{
    if ((rp & 3) == 3) { gpu_set_AF(s, v); return; }
    gpu_rp_w(s, rp, v);
}

/* ALU helpers */
__device__ static void gpu_alu_add(GBStateGPU *s, uint8_t v, int cy)
{
    int r = (int)s->A + (int)v + cy;
    GPU_SET_H(s, ((s->A & 0xF) + (v & 0xF) + cy) > 0xF);
    GPU_SET_C(s, r > 0xFF);
    s->A = (uint8_t)r;
    GPU_SET_Z(s, s->A == 0);
    GPU_SET_N(s, 0);
}
__device__ static void gpu_alu_sub(GBStateGPU *s, uint8_t v, int cy, int store)
{
    int r = (int)s->A - (int)v - cy;
    GPU_SET_H(s, ((int)(s->A & 0xF) - (int)(v & 0xF) - cy) < 0);
    GPU_SET_C(s, r < 0);
    if (store) s->A = (uint8_t)r;
    GPU_SET_Z(s, (uint8_t)r == 0);
    GPU_SET_N(s, 1);
}
__device__ static void gpu_alu_and(GBStateGPU *s, uint8_t v)
{
    s->A &= v;
    s->F = GPU_FLAG_H;
    GPU_SET_Z(s, s->A == 0);
}
__device__ static void gpu_alu_xor(GBStateGPU *s, uint8_t v)
{
    s->A ^= v;
    s->F = 0;
    GPU_SET_Z(s, s->A == 0);
}
__device__ static void gpu_alu_or(GBStateGPU *s, uint8_t v)
{
    s->A |= v;
    s->F = 0;
    GPU_SET_Z(s, s->A == 0);
}

__device__ static int gpu_cc(const GBStateGPU *s, int cc)
{
    switch (cc & 3) {
        case 0: return !GPU_GET_Z(s);
        case 1: return  GPU_GET_Z(s);
        case 2: return !GPU_GET_C(s);
        case 3: return  GPU_GET_C(s);
    }
    return 0;
}

/* Full SM83 cpu_step equivalent (no trace, no halt-bug) */
__device__ static int gpu_cpu_step_fallback(
    GBStateGPU   *s,
    uint8_t      *vram,
    uint8_t      *wram,
    uint8_t      *ext_ram,
    const uint8_t *rom, uint32_t rom_size, uint32_t ext_ram_size)
{
    /* EI delay */
    int ime_this = s->ime_pending;

    uint8_t  op = gpu_mem_read(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size, s->PC++);

    /* HALT bug (simplified) */
    if (s->halt_bug) { s->PC--; s->halt_bug = 0; }

    int cycles = 4;

    switch (op) {
    case 0x00: /* NOP */ break;

    /* ── LD rp, nn ── */
    case 0x01: case 0x11: case 0x21: case 0x31: {
        uint8_t lo = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, s->PC++);
        uint8_t hi = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, s->PC++);
        gpu_rp_w(s, (op>>4)&3, (uint16_t)((hi<<8)|lo));
        cycles = 12; break;
    }
    /* ── LD [rp], A / LD A, [rp] ── */
    case 0x02: gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_BC(s), s->A); cycles=8; break;
    case 0x12: gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_DE(s), s->A); cycles=8; break;
    case 0x0A: s->A = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_BC(s)); cycles=8; break;
    case 0x1A: s->A = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_DE(s)); cycles=8; break;
    case 0x22: gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_HL(s), s->A); gpu_set_HL(s,(uint16_t)(gpu_HL(s)+1)); cycles=8; break;
    case 0x32: gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_HL(s), s->A); gpu_set_HL(s,(uint16_t)(gpu_HL(s)-1)); cycles=8; break;
    case 0x2A: s->A = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_HL(s)); gpu_set_HL(s,(uint16_t)(gpu_HL(s)+1)); cycles=8; break;
    case 0x3A: s->A = gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size, gpu_HL(s)); gpu_set_HL(s,(uint16_t)(gpu_HL(s)-1)); cycles=8; break;

    /* ── INC/DEC rp ── */
    case 0x03: case 0x13: case 0x23: case 0x33:
        gpu_rp_w(s,(op>>4)&3, (uint16_t)(gpu_rp(s,(op>>4)&3)+1)); cycles=8; break;
    case 0x0B: case 0x1B: case 0x2B: case 0x3B:
        gpu_rp_w(s,(op>>4)&3, (uint16_t)(gpu_rp(s,(op>>4)&3)-1)); cycles=8; break;

    /* ── INC r8 ── */
    case 0x04: case 0x0C: case 0x14: case 0x1C:
    case 0x24: case 0x2C: case 0x34: case 0x3C: {
        int r = (op>>3)&7;
        uint8_t v = gpu_reg_r(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r);
        uint8_t n = (uint8_t)(v+1);
        GPU_SET_Z(s,n==0); GPU_SET_N(s,0); GPU_SET_H(s,(v&0xF)==0xF);
        gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,n);
        cycles=(r==6)?12:4; break;
    }
    /* ── DEC r8 ── */
    case 0x05: case 0x0D: case 0x15: case 0x1D:
    case 0x25: case 0x2D: case 0x35: case 0x3D: {
        int r = (op>>3)&7;
        uint8_t v = gpu_reg_r(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r);
        uint8_t n = (uint8_t)(v-1);
        GPU_SET_Z(s,n==0); GPU_SET_N(s,1); GPU_SET_H(s,(v&0xF)==0);
        gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,n);
        cycles=(r==6)?12:4; break;
    }
    /* ── LD r8, n8 ── */
    case 0x06: case 0x0E: case 0x16: case 0x1E:
    case 0x26: case 0x2E: case 0x36: case 0x3E: {
        int r=(op>>3)&7;
        uint8_t n=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,n);
        cycles=(r==6)?12:8; break;
    }
    /* ── Rotates on A ── */
    case 0x07: { uint8_t b=s->A>>7; s->A=(uint8_t)((s->A<<1)|b); s->F=0; GPU_SET_C(s,b); break; } /* RLCA */
    case 0x0F: { uint8_t b=s->A&1;  s->A=(uint8_t)((s->A>>1)|(b<<7)); s->F=0; GPU_SET_C(s,b); break; } /* RRCA */
    case 0x17: { uint8_t cf=GPU_GET_C(s)?1:0; uint8_t b=s->A>>7; s->A=(uint8_t)((s->A<<1)|cf); s->F=0; GPU_SET_C(s,b); break; } /* RLA */
    case 0x1F: { uint8_t cf=GPU_GET_C(s)?1:0; uint8_t b=s->A&1; s->A=(uint8_t)((s->A>>1)|(cf<<7)); s->F=0; GPU_SET_C(s,b); break; } /* RRA */

    /* ── JR ── */
    case 0x18: { int8_t e=(int8_t)gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++); s->PC=(uint16_t)(s->PC+e); cycles=12; break; }
    case 0x20: case 0x28: case 0x30: case 0x38: {
        int8_t e=(int8_t)gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        if (gpu_cc(s,(op>>3)&3)) { s->PC=(uint16_t)(s->PC+e); cycles=12; }
        else cycles=8;
        break;
    }

    /* ── LD [nn], SP / LD HL, SP+e8 ── */
    case 0x08: {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint16_t a=(uint16_t)((hi<<8)|lo);
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,a,(uint8_t)(s->SP&0xFF));
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(a+1),(uint8_t)(s->SP>>8));
        cycles=20; break;
    }
    /* ── ADD HL, rp ── */
    case 0x09: case 0x19: case 0x29: case 0x39: {
        uint16_t v=gpu_rp(s,(op>>4)&3); uint16_t hl=gpu_HL(s);
        uint32_t r=(uint32_t)hl+v;
        GPU_SET_N(s,0); GPU_SET_H(s,((hl&0xFFF)+(v&0xFFF))>0xFFF); GPU_SET_C(s,r>0xFFFF);
        gpu_set_HL(s,(uint16_t)r); cycles=8; break;
    }
    /* ── STOP ── */
    case 0x10: s->PC++; cycles=4; break;
    /* ── DAA ── */
    case 0x27: s->A=gpu_exec_daa(s); break;
    /* ── CPL ── */
    case 0x2F: s->A=~s->A; GPU_SET_N(s,1); GPU_SET_H(s,1); break;
    /* ── SCF / CCF ── */
    case 0x37: GPU_SET_N(s,0); GPU_SET_H(s,0); GPU_SET_C(s,1); break;
    case 0x3F: GPU_SET_N(s,0); GPU_SET_H(s,0); GPU_SET_C(s,!GPU_GET_C(s)); break;
    /* ── HALT ── */
    case 0x76: {
        uint8_t pending2 = s->io[0x0F] & s->IE & 0x1F;
        if (s->IME) { s->halted=1; }
        else if (!pending2) { s->halted=1; }
        else { s->halt_bug=1; } /* halt bug */
        break;
    }
    /* ── LD r8, r8 (0x40–0x7F) ── */
    default:
    if (op >= 0x40 && op <= 0x7F) {
        int dst=(op>>3)&7, src=op&7;
        uint8_t v=gpu_reg_r(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,src);
        gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,dst,v);
        cycles=(dst==6||src==6)?8:4; break;
    }
    /* ── ALU A, r8 (0x80–0xBF) ── */
    else if (op >= 0x80 && op <= 0xBF) {
        uint8_t v=gpu_reg_r(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,op&7);
        int alu=(op>>3)&7;
        switch(alu){
            case 0: gpu_alu_add(s,v,0); break;
            case 1: gpu_alu_add(s,v,GPU_GET_C(s)?1:0); break;
            case 2: gpu_alu_sub(s,v,0,1); break;
            case 3: gpu_alu_sub(s,v,GPU_GET_C(s)?1:0,1); break;
            case 4: gpu_alu_and(s,v); break;
            case 5: gpu_alu_xor(s,v); break;
            case 6: gpu_alu_or(s,v); break;
            case 7: gpu_alu_sub(s,v,0,0); break; /* CP */
        }
        cycles=((op&7)==6)?8:4; break;
    }
    /* ── RET cc / RET / RETI ── */
    else if (op==0xC0||op==0xC8||op==0xD0||op==0xD8) {
        if (gpu_cc(s,(op>>3)&3)) {
            uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
            uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
            s->PC=(uint16_t)((hi<<8)|lo); cycles=20;
        } else cycles=8;
        break;
    }
    else if (op==0xC9) { /* RET */
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        s->PC=(uint16_t)((hi<<8)|lo); cycles=16; break;
    }
    else if (op==0xD9) { /* RETI */
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        s->PC=(uint16_t)((hi<<8)|lo); s->IME=1; cycles=16; break;
    }
    /* ── POP rp2 ── */
    else if (op==0xC1||op==0xD1||op==0xE1||op==0xF1) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP++);
        gpu_rp2_w(s,(op>>4)&3,(uint16_t)((hi<<8)|lo)); cycles=12; break;
    }
    /* ── JP cc, nn ── */
    else if (op==0xC2||op==0xCA||op==0xD2||op==0xDA) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        if (gpu_cc(s,(op>>3)&3)) { s->PC=(uint16_t)((hi<<8)|lo); cycles=16; }
        else cycles=12;
        break;
    }
    /* ── JP nn ── */
    else if (op==0xC3) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        s->PC=(uint16_t)((hi<<8)|lo); cycles=16; break;
    }
    /* ── CALL cc, nn ── */
    else if (op==0xC4||op==0xCC||op==0xD4||op==0xDC) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        if (gpu_cc(s,(op>>3)&3)) {
            s->SP-=2;
            gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(s->SP+1),(uint8_t)(s->PC>>8));
            gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP,(uint8_t)(s->PC&0xFF));
            s->PC=(uint16_t)((hi<<8)|lo); cycles=24;
        } else cycles=12;
        break;
    }
    /* ── PUSH rp2 ── */
    else if (op==0xC5||op==0xD5||op==0xE5||op==0xF5) {
        uint16_t v=gpu_rp2(s,(op>>4)&3);
        s->SP-=2;
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(s->SP+1),(uint8_t)(v>>8));
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP,(uint8_t)(v&0xFF));
        cycles=16; break;
    }
    /* ── ALU A, n8 (0xC6..0xFE with low nibble 6 or E) ── */
    else if ((op&0xC7)==0xC6) {
        uint8_t v=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        int alu=(op>>3)&7;
        switch(alu){
            case 0: gpu_alu_add(s,v,0); break;
            case 1: gpu_alu_add(s,v,GPU_GET_C(s)?1:0); break;
            case 2: gpu_alu_sub(s,v,0,1); break;
            case 3: gpu_alu_sub(s,v,GPU_GET_C(s)?1:0,1); break;
            case 4: gpu_alu_and(s,v); break;
            case 5: gpu_alu_xor(s,v); break;
            case 6: gpu_alu_or(s,v); break;
            case 7: gpu_alu_sub(s,v,0,0); break; /* CP n8 */
        }
        cycles=8; break;
    }
    /* ── RST ── */
    else if ((op&0xC7)==0xC7) {
        uint16_t ra=s->PC; s->SP-=2;
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(s->SP+1),(uint8_t)(ra>>8));
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP,(uint8_t)(ra&0xFF));
        s->PC=(uint16_t)(op&0x38); cycles=16; break;
    }
    /* ── CALL nn ── */
    else if (op==0xCD) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        s->SP-=2;
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(s->SP+1),(uint8_t)(s->PC>>8));
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->SP,(uint8_t)(s->PC&0xFF));
        s->PC=(uint16_t)((hi<<8)|lo); cycles=24; break;
    }
    /* ── LDH [a8], A / LDH A, [a8] ── */
    else if (op==0xE0) {
        uint8_t a=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(0xFF00|a),s->A);
        cycles=12; break;
    }
    else if (op==0xF0) {
        uint8_t a=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        s->A=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(0xFF00|a));
        cycles=12; break;
    }
    /* ── LD [C], A / LD A, [C] ── */
    else if (op==0xE2) {
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(0xFF00|s->C),s->A);
        cycles=8; break;
    }
    else if (op==0xF2) {
        s->A=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)(0xFF00|s->C));
        cycles=8; break;
    }
    /* ── ADD SP, e8 ── */
    else if (op==0xE8) {
        int8_t e=(int8_t)gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint32_t sp=s->SP;
        GPU_SET_Z(s,0); GPU_SET_N(s,0);
        GPU_SET_H(s,((sp&0xF)+(uint32_t)((uint8_t)e&0xF))>0xF);
        GPU_SET_C(s,((sp&0xFF)+(uint32_t)(uint8_t)e)>0xFF);
        s->SP=(uint16_t)((int32_t)(int16_t)s->SP+e);
        cycles=16; break;
    }
    /* ── LD HL, SP+e8 ── */
    else if (op==0xF8) {
        int8_t e=(int8_t)gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint32_t sp=s->SP;
        GPU_SET_Z(s,0); GPU_SET_N(s,0);
        GPU_SET_H(s,((sp&0xF)+(uint32_t)((uint8_t)e&0xF))>0xF);
        GPU_SET_C(s,((sp&0xFF)+(uint32_t)(uint8_t)e)>0xFF);
        gpu_set_HL(s,(uint16_t)((int32_t)(int16_t)s->SP+e));
        cycles=12; break;
    }
    /* ── JP [HL] ── */
    else if (op==0xE9) { s->PC=gpu_HL(s); cycles=4; break; }
    /* ── LD SP, HL ── */
    else if (op==0xF9) { s->SP=gpu_HL(s); cycles=8; break; }
    /* ── LD [nn], A ── */
    else if (op==0xEA) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        gpu_mem_write(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)((hi<<8)|lo),s->A);
        cycles=16; break;
    }
    /* ── LD A, [nn] ── */
    else if (op==0xFA) {
        uint8_t lo=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        uint8_t hi=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        s->A=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,(uint16_t)((hi<<8)|lo));
        cycles=16; break;
    }
    /* ── DI / EI ── */
    else if (op==0xF3) { s->IME=0; s->ime_pending=0; cycles=4; break; }
    else if (op==0xFB) { ime_this=1; /* apply below */ cycles=4; break; }
    /* ── CB prefix ── */
    else if (op==0xCB) {
        uint8_t cb=gpu_mem_read(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,s->PC++);
        int r=cb&7;
        uint8_t v=gpu_reg_r(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r);
        uint8_t res=v;
        int bit=(cb>>3)&7;
        switch(cb>>6) {
            case 0: /* rot/shift */
                switch(bit) {
                    case 0: { uint8_t b=v>>7; res=(uint8_t)((v<<1)|b); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* RLC */
                    case 1: { uint8_t b=v&1;  res=(uint8_t)((v>>1)|(b<<7)); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* RRC */
                    case 2: { uint8_t cf=GPU_GET_C(s)?1:0; uint8_t b=v>>7; res=(uint8_t)((v<<1)|cf); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* RL */
                    case 3: { uint8_t cf=GPU_GET_C(s)?1:0; uint8_t b=v&1; res=(uint8_t)((v>>1)|(cf<<7)); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* RR */
                    case 4: { uint8_t b=v>>7; res=(uint8_t)(v<<1); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* SLA */
                    case 5: { uint8_t b=v&1; res=(uint8_t)((int8_t)v>>1); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* SRA */
                    case 6: res=(uint8_t)(((v&0xF)<<4)|((v>>4)&0xF)); s->F=0; GPU_SET_Z(s,res==0); break; /* SWAP */
                    case 7: { uint8_t b=v&1; res=(uint8_t)(v>>1); s->F=0; GPU_SET_Z(s,res==0); GPU_SET_C(s,b); break; } /* SRL */
                }
                gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,res);
                break;
            case 1: /* BIT */
                GPU_SET_Z(s,((v>>bit)&1)==0); GPU_SET_N(s,0); GPU_SET_H(s,1);
                break;
            case 2: /* RES */
                res=(uint8_t)(v&~(1<<bit));
                gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,res);
                break;
            case 3: /* SET */
                res=(uint8_t)(v|(1<<bit));
                gpu_reg_w(s,vram,wram,ext_ram,rom,rom_size,ext_ram_size,r,res);
                break;
        }
        cycles=(r==6)?16:8; break;
    }
    /* ── Unknown ── */
    else { /* treat as NOP */ break; }
    break; /* end outer default */
    } /* switch op */

    /* EI delay: apply IME after next instruction */
    if (ime_this) {
        s->IME = 1;
        s->ime_pending = 0;
    }

    s->cycles += (uint64_t)cycles;
    return cycles;
}

/* ============================================================
 * Device: One IR step (one block or interrupt service)
 * ============================================================ */

/* ============================================================
 * O(1) hash table lookup
 *   Knuth multiplicative hash maps (bank<<16)|PC → slot in a
 *   131072-entry open-addressing table (~34% load for 44k blocks).
 * ============================================================ */

__device__ __forceinline__ static int gpu_hash_find(
    const BlockHashSlot *hash, uint32_t key)
{
    uint32_t h = (key * 2654435761u) >> (32 - GPU_BLOCK_HASH_BITS);
    while (true) {
        uint32_t k = hash[h].key;
        if (k == key)                    return hash[h].idx;
        if (k == GPU_BLOCK_HASH_EMPTY)   return -1;
        h = (h + 1) & (GPU_BLOCK_HASH_SIZE - 1);
    }
}

/* ============================================================
 * Device: gpu_ir_step
 *   • Hash-table lookup (O(1)) instead of binary search
 *   • Block chaining: execute up to GPU_CHAIN_DEPTH consecutive
 *     blocks per call, amortising ppu/timer/dma overhead.
 *   • Calls gpu_ppu_step/gpu_timer_step/gpu_dma_tick after every
 *     block in the chain to preserve correct interrupt timing.
 * ============================================================ */

__device__ static void gpu_ir_step(
    GBStateGPU          *s,
    uint8_t             *vram,
    uint8_t             *wram,
    uint8_t             *ext_ram,
    const uint8_t       *rom,
    uint32_t             rom_size,
    uint32_t             ext_ram_size,
    const IRBlockInfoGPU *blocks,
    const IRInsn        *insns,
    const BlockHashSlot *hash,
    uint32_t            *d_profile)   /* [n_blocks] atomic counters; NULL = off */
{
    /* EI delay: IME takes effect before this step's interrupt check */
    if (s->ime_pending) { s->IME = 1; s->ime_pending = 0; }

    uint8_t pending = s->io[0x0F] & s->IE & 0x1F;

    /* ── Halt / interrupt service ─────────────────────────────── */
    if (s->halted) {
        if (!pending) {
            /* HALT fast-forward: skip to next interrupt event in one shot.
             * Without this we'd spin at 4 cycles/call for up to 17,500
             * iterations per frame while the game waits for VBlank. */
            int skip = gpu_cycles_to_wake(s);
            gpu_ppu_step(s, skip);
            gpu_timer_step(s, skip);
            gpu_dma_tick(s, ext_ram, rom, rom_size, ext_ram_size, skip);
            return;
        }
        s->halted = 0;
        if (s->IME) {
            int c = gpu_service_interrupt(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size);
            gpu_ppu_step(s, c);
            gpu_timer_step(s, c);
            gpu_dma_tick(s, ext_ram, rom, rom_size, ext_ram_size, c);
            return;
        }
        /* IME=0: wake but don't service — fall through to block fetch */
    } else if (s->IME && pending) {
        int c = gpu_service_interrupt(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size);
        gpu_ppu_step(s, c);
        gpu_timer_step(s, c);
        gpu_dma_tick(s, ext_ram, rom, rom_size, ext_ram_size, c);
        return;
    }

    /* ── Block execution with chaining ──────────────────────────
     * After each block we call ppu/timer/dma so that interrupt bits
     * (VBlank, STAT, timer) are set before the next block is fetched.
     * This exactly matches the non-chaining behaviour while keeping
     * intermediate state in GPU registers across the loop body. */
    for (int depth = 0; depth < GPU_CHAIN_DEPTH; depth++) {
        int bank  = (s->PC < 0x4000u) ? 0 : (int)s->rom_bank;
        uint32_t key = ((uint32_t)bank << 16) | (uint32_t)s->PC;

        /* ── JIT hot-trace dispatch (fastest path) ── */
        int trace_id = trace_lookup(key);
        if (trace_id >= 0) {
            /* jit_exec_trace ticks PPU/timer/DMA per block internally */
            jit_exec_trace(trace_id, s, vram, wram, ext_ram, rom,
                           rom_size, ext_ram_size);
            return; /* trace covers multiple blocks; recheck interrupts next entry */
        }

        int found = gpu_hash_find(hash, key);

        /* ── Optional block-hit profiling ── */
        if (d_profile != nullptr && found >= 0)
            atomicAdd(&d_profile[found], 1u);

        int cyc;
        if (found < 0) {
            cyc = gpu_cpu_step_fallback(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size);
            s->fallback_count++;
            s->fallback_cycles += (uint64_t)cyc;
        } else {
            cyc = gpu_exec_block(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                                 blocks + found, insns);
        }

        /* Advance subsystems — may set interrupt bits */
        gpu_ppu_step(s, cyc);
        gpu_timer_step(s, cyc);
        gpu_dma_tick(s, ext_ram, rom, rom_size, ext_ram_size, cyc);

        /* Break chain if:
         *  • fallback ran (unknown code; might have changed ROM banks)
         *  • an interrupt is now pending (service next call)
         *  • entered halt state
         *  • EI was executed (ime_pending, apply at next call's start) */
        if (found < 0 ||
            (s->io[0x0F] & s->IE & 0x1F) ||
            s->halted ||
            s->ime_pending)
            return;
    }
}

/* ============================================================
 * Sorted dispatch helpers
 * ============================================================ */

/* Fill keys[tid] = (rom_bank<<16)|PC — used as sort key for warp coherence. */
__global__ void gpu_compute_block_keys(
    const GBStateGPU *states, uint32_t *keys, int n_states)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_states) return;
    const GBStateGPU *s = &states[tid];
    int bank = (s->PC < 0x4000u) ? 0 : (int)s->rom_bank;
    keys[tid] = ((uint32_t)bank << 16) | (uint32_t)s->PC;
}

/* Initialize indices[tid] = tid (iota). */
__global__ void gpu_iota(int *indices, int n)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) indices[tid] = tid;
}

/* ============================================================
 * Kernel: run N frames — sorted dispatch
 *
 * Thread `tid` operates on state `sorted_indices[tid]` so that
 * adjacent threads in a warp handle states at the same block key,
 * eliminating intra-warp branch divergence.
 * ============================================================ */

__global__ void gpu_run_frames_sorted_kernel(
    GBStateGPU          *states,
    uint8_t             *d_vrams,
    uint8_t             *d_wrams,
    uint8_t             *ext_rams,
    const uint8_t       *rom,
    uint32_t             rom_size,
    uint32_t             ext_ram_size,
    const IRBlockInfoGPU *blocks,
    int                  n_blocks,
    const IRInsn        *insns,
    const uint8_t       *actions,
    const int           *sorted_indices,
    const BlockHashSlot *hash,
    uint32_t            *d_profile,     /* NULL = off */
    int                  n_states,
    int                  n_frames)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_states) return;

    int         idx     = sorted_indices[tid];
    GBStateGPU *s       = &states[idx];
    uint8_t    *vram    = d_vrams + (uint64_t)idx * GPU_VRAM_SIZE;
    uint8_t    *wram    = d_wrams + (uint64_t)idx * GPU_WRAM_SIZE;
    uint8_t    *ext_ram = (ext_ram_size > 0)
                          ? (ext_rams + (uint64_t)idx * ext_ram_size)
                          : nullptr;

    s->joypad_action = actions[idx];

    uint32_t target = s->ppu_frame + (uint32_t)n_frames;

    while (s->ppu_frame < target) {
        gpu_ir_step(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                    blocks, insns, hash, d_profile);
    }
}

/* ============================================================
 * Kernel: run N frames (legacy unsorted — kept for testing)
 * ============================================================ */

__global__ void gpu_run_frames_kernel(
    GBStateGPU          *states,
    uint8_t             *d_vrams,
    uint8_t             *d_wrams,
    uint8_t             *ext_rams,
    const uint8_t       *rom,
    uint32_t             rom_size,
    uint32_t             ext_ram_size,
    const IRBlockInfoGPU *blocks,
    int                  n_blocks,
    const IRInsn        *insns,
    const BlockHashSlot *hash,
    uint32_t            *d_profile,
    int                  n_states,
    int                  n_frames)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_states) return;

    GBStateGPU *s       = &states[tid];
    uint8_t    *vram    = d_vrams + (uint64_t)tid * GPU_VRAM_SIZE;
    uint8_t    *wram    = d_wrams + (uint64_t)tid * GPU_WRAM_SIZE;
    uint8_t    *ext_ram = (ext_ram_size > 0)
                          ? (ext_rams + (uint64_t)tid * ext_ram_size)
                          : nullptr;

    uint32_t target = s->ppu_frame + (uint32_t)n_frames;

    while (s->ppu_frame < target) {
        gpu_ir_step(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                    blocks, insns, hash, d_profile);
    }
}

/* ============================================================
 * Host: conversion helpers
 * ============================================================ */

void gb_to_gpu_state(const GB *gb, GBStateGPU *g)
{
    memset(g, 0, sizeof(*g));

    /* CPU */
    g->A = gb->cpu.A;  g->F = gb->cpu.F;
    g->B = gb->cpu.B;  g->C = gb->cpu.C;
    g->D = gb->cpu.D;  g->E = gb->cpu.E;
    g->H = gb->cpu.H;  g->L = gb->cpu.L;
    g->SP     = gb->cpu.SP;
    g->PC     = gb->cpu.PC;
    g->IME          = gb->cpu.IME         ? 1 : 0;
    g->ime_pending  = gb->cpu.ime_pending ? 1 : 0;
    g->halted       = gb->cpu.halted      ? 1 : 0;
    g->halt_bug     = gb->cpu.halt_bug    ? 1 : 0;
    g->cycles       = gb->cpu.cycles;

    /* MBC */
    g->mbc_type   = (uint8_t)gb->mem.mbc_type;
    g->rom_bank   = gb->mem.rom_bank;
    g->ram_bank   = gb->mem.ram_bank;
    g->ram_enabled = gb->mem.ram_enabled ? 1 : 0;
    g->mbc1_mode  = gb->mem.mbc1_mode;

    g->IE = gb->mem.IE;

    /* DMA — dma_index is repurposed on GPU as T-cycles remaining.
     * CPU stores byte index (0-159); convert: remaining = (160-byte_idx)*4 */
    g->dma_active = gb->mem.dma_active ? 1 : 0;
    g->dma_src    = gb->mem.dma_src;
    g->dma_index  = gb->mem.dma_active
                    ? (160 - gb->mem.dma_index) * 4
                    : 0;

    /* PPU */
    g->ppu_scanline = gb->ppu.scanline;
    g->ppu_dot      = gb->ppu.dot;
    g->ppu_mode     = (int32_t)gb->ppu.mode;
    g->ppu_frame    = gb->ppu.frame;

    /* Timer */
    g->timer_div             = gb->timer.div_internal;
    g->timer_tima            = gb->timer.tima;
    g->timer_tma             = gb->timer.tma;
    g->timer_tac             = gb->timer.tac;
    g->timer_overflow        = gb->timer.tima_overflow ? 1 : 0;
    g->timer_overflow_delay  = gb->timer.overflow_delay;

    /* IO + memory (vram and wram NOT inline — stored in d_vrams/d_wrams, copied separately) */
    memcpy(g->io,   gb->mem.io,   sizeof(g->io));
    memcpy(g->hram, gb->mem.hram, sizeof(g->hram));
    memcpy(g->oam,  gb->mem.oam,  sizeof(g->oam));
}

void gpu_state_to_gb(const GBStateGPU *g, GB *gb)
{
    /* CPU */
    gb->cpu.A = g->A;  gb->cpu.F = g->F;
    gb->cpu.B = g->B;  gb->cpu.C = g->C;
    gb->cpu.D = g->D;  gb->cpu.E = g->E;
    gb->cpu.H = g->H;  gb->cpu.L = g->L;
    gb->cpu.SP          = g->SP;
    gb->cpu.PC          = g->PC;
    gb->cpu.IME         = g->IME         != 0;
    gb->cpu.ime_pending = g->ime_pending != 0;
    gb->cpu.halted      = g->halted      != 0;
    gb->cpu.halt_bug    = g->halt_bug    != 0;
    gb->cpu.cycles      = g->cycles;

    /* MBC */
    gb->mem.mbc_type    = (MBCType)g->mbc_type;
    gb->mem.rom_bank    = g->rom_bank;
    gb->mem.ram_bank    = g->ram_bank;
    gb->mem.ram_enabled = g->ram_enabled != 0;
    gb->mem.mbc1_mode   = g->mbc1_mode;

    gb->mem.IE = g->IE;

    /* DMA — convert GPU T-cycles-remaining back to CPU byte index */
    gb->mem.dma_active = g->dma_active != 0;
    gb->mem.dma_src    = g->dma_src;
    {
        int remaining_tcyc = (g->dma_active && g->dma_index > 0) ? g->dma_index : 0;
        int byte_idx = 160 - (remaining_tcyc / 4);
        if (byte_idx < 0)   byte_idx = 0;
        if (byte_idx > 160) byte_idx = 160;
        gb->mem.dma_index = byte_idx;
    }

    /* PPU */
    gb->ppu.scanline = g->ppu_scanline;
    gb->ppu.dot      = g->ppu_dot;
    gb->ppu.mode     = (PPUMode)g->ppu_mode;
    gb->ppu.frame    = g->ppu_frame;

    /* Timer */
    gb->timer.div_internal   = g->timer_div;
    gb->timer.tima            = g->timer_tima;
    gb->timer.tma             = g->timer_tma;
    gb->timer.tac             = g->timer_tac;
    gb->timer.tima_overflow   = g->timer_overflow != 0;
    gb->timer.overflow_delay  = g->timer_overflow_delay;

    /* IO + memory (vram+wram NOT inline — copied separately in gpu_batch_sync_to_host) */
    memcpy(gb->mem.io,   g->io,   sizeof(g->io));
    memcpy(gb->mem.hram, g->hram, sizeof(g->hram));
    memcpy(gb->mem.oam,  g->oam,  sizeof(g->oam));
}

/* ============================================================
 * Host: IR serialisation (lift all CFG blocks)
 * ============================================================ */

typedef struct { uint32_t key; int orig_idx; } SortKI;

static int cmp_sort_ki(const void *a, const void *b)
{
    const SortKI *x = (const SortKI *)a;
    const SortKI *y = (const SortKI *)b;
    return (x->key > y->key) - (x->key < y->key);
}

static int gpu_serialize_ir(
    const CFG      *cfg,
    IRBlockInfoGPU **out_blocks,
    int             *out_n_blocks,
    IRInsn         **out_insns,
    int             *out_n_insns)
{
    int n = cfg->n_blocks;

    /* Lift all blocks */
    IRBlock **lifted = (IRBlock **)calloc((size_t)n, sizeof(IRBlock *));
    if (!lifted) return -1;

    int total_insns = 0;
    for (int i = 0; i < n; i++) {
        lifted[i] = ir_lift_block(&cfg->blocks[i]);
        if (lifted[i]) total_insns += lifted[i]->n_insns;
    }

    /* Sort blocks by key */
    SortKI *ki = (SortKI *)malloc((size_t)n * sizeof(SortKI));
    if (!ki) { free(lifted); return -1; }
    for (int i = 0; i < n; i++) {
        ki[i].key      = ((uint32_t)cfg->blocks[i].bank << 16)
                       | cfg->blocks[i].entry;
        ki[i].orig_idx = i;
    }
    qsort(ki, (size_t)n, sizeof(SortKI), cmp_sort_ki);

    /* Allocate output arrays */
    IRBlockInfoGPU *blocks = (IRBlockInfoGPU *)malloc(
        (size_t)n * sizeof(IRBlockInfoGPU));
    IRInsn *insns = (IRInsn *)malloc((size_t)total_insns * sizeof(IRInsn));
    if (!blocks || !insns) {
        free(blocks); free(insns); free(ki); free(lifted); return -1;
    }

    /* Fill in sorted order */
    int insn_cursor = 0;
    for (int si = 0; si < n; si++) {
        int i = ki[si].orig_idx;
        IRBlock *ib = lifted[i];

        blocks[si].key          = ki[si].key;
        blocks[si].exit_type    = (int)cfg->blocks[i].exit_type;
        blocks[si].succs[0]     = (ib && ib->n_succs > 0) ? ib->succs[0] : 0;
        blocks[si].succs[1]     = (ib && ib->n_succs > 1) ? ib->succs[1] : 0;
        blocks[si].insn_offset  = insn_cursor;

        if (ib) {
            blocks[si].n_insns      = ib->n_insns;
            blocks[si].n_tmps       = ib->n_tmps;
            blocks[si].taken_cycles = ib->taken_cycles;
            blocks[si].fall_cycles  = ib->fall_cycles;
            memcpy(insns + insn_cursor, ib->insns,
                   (size_t)ib->n_insns * sizeof(IRInsn));
            insn_cursor += ib->n_insns;
        } else {
            blocks[si].n_insns      = 0;
            blocks[si].n_tmps       = 0;
            blocks[si].taken_cycles = 4;
            blocks[si].fall_cycles  = 4;
        }
    }

    /* Free lifted blocks */
    for (int i = 0; i < n; i++)
        if (lifted[i]) ir_free_block(lifted[i]);
    free(lifted);
    free(ki);

    *out_blocks   = blocks;
    *out_n_blocks = n;
    *out_insns    = insns;
    *out_n_insns  = insn_cursor;
    return 0;
}

/* ============================================================
 * Host API
 * ============================================================ */

GPUBatch *gpu_batch_create(
    const CFG     *cfg,
    const GB      *host_states,
    const uint8_t *host_ext_rams,
    int            n_states,
    uint32_t       rom_size,
    uint32_t       ext_ram_size)
{
    GPUBatch *b = (GPUBatch *)calloc(1, sizeof(GPUBatch));
    if (!b) return NULL;

    b->n_states    = n_states;
    b->rom_size    = rom_size;
    b->ext_ram_size = ext_ram_size;

    /* ── Serialise all IR blocks ── */
    fprintf(stderr, "GPU: lifting all %d CFG blocks...\n", cfg->n_blocks);
    IRBlockInfoGPU *h_blocks = NULL;
    IRInsn         *h_insns  = NULL;
    if (gpu_serialize_ir(cfg, &h_blocks, &b->n_blocks,
                         &h_insns,  &b->n_insns) != 0) {
        free(b); return NULL;
    }
    fprintf(stderr, "GPU: %d blocks, %d IR insns serialised.\n",
            b->n_blocks, b->n_insns);

    /* ── Convert host GB states to GBStateGPU ── */
    GBStateGPU *h_states = (GBStateGPU *)malloc(
        (size_t)n_states * sizeof(GBStateGPU));
    /* Host VRAM/WRAM buffers: n_states × 8 KB each (extracted from GBStateGPU) */
    uint8_t *h_vrams = (uint8_t *)malloc((size_t)n_states * GPU_VRAM_SIZE);
    uint8_t *h_wrams = (uint8_t *)malloc((size_t)n_states * GPU_WRAM_SIZE);
    if (!h_states || !h_vrams || !h_wrams) {
        free(h_states); free(h_vrams); free(h_wrams);
        free(h_blocks); free(h_insns); free(b); return NULL;
    }
    for (int i = 0; i < n_states; i++) {
        gb_to_gpu_state(&host_states[i], &h_states[i]);
        memcpy(h_vrams + (size_t)i * GPU_VRAM_SIZE,
               host_states[i].mem.vram, GPU_VRAM_SIZE);
        memcpy(h_wrams + (size_t)i * GPU_WRAM_SIZE,
               host_states[i].mem.wram, GPU_WRAM_SIZE);
    }

    /* ── Allocate & upload to device ── */
    /* States */
    CUDA_CHECK(cudaMalloc(&b->d_states,
        (size_t)n_states * sizeof(GBStateGPU)));
    CUDA_CHECK(cudaMemcpy(b->d_states, h_states,
        (size_t)n_states * sizeof(GBStateGPU), cudaMemcpyHostToDevice));
    free(h_states);

    /* VRAMs (per-state 8 KB arrays, extracted from GBStateGPU) */
    CUDA_CHECK(cudaMalloc(&b->d_vrams, (size_t)n_states * GPU_VRAM_SIZE));
    CUDA_CHECK(cudaMemcpy(b->d_vrams, h_vrams,
        (size_t)n_states * GPU_VRAM_SIZE, cudaMemcpyHostToDevice));
    free(h_vrams);

    /* WRAMs (per-state 8 KB arrays, extracted from GBStateGPU) */
    CUDA_CHECK(cudaMalloc(&b->d_wrams, (size_t)n_states * GPU_WRAM_SIZE));
    CUDA_CHECK(cudaMemcpy(b->d_wrams, h_wrams,
        (size_t)n_states * GPU_WRAM_SIZE, cudaMemcpyHostToDevice));
    free(h_wrams);

    /* ROM (shared, read-only) */
    uint8_t *d_rom_rw = NULL;
    CUDA_CHECK(cudaMalloc(&d_rom_rw, rom_size));
    CUDA_CHECK(cudaMemcpy(d_rom_rw, host_states[0].mem.rom,
        rom_size, cudaMemcpyHostToDevice));
    b->d_rom = d_rom_rw;

    /* ext_ram (per-instance) */
    if (ext_ram_size > 0) {
        CUDA_CHECK(cudaMalloc(&b->d_ext_rams,
            (size_t)n_states * ext_ram_size));
        CUDA_CHECK(cudaMemcpy(b->d_ext_rams, host_ext_rams,
            (size_t)n_states * ext_ram_size, cudaMemcpyHostToDevice));
    }

    /* IR blocks */
    CUDA_CHECK(cudaMalloc(&b->d_blocks,
        (size_t)b->n_blocks * sizeof(IRBlockInfoGPU)));
    CUDA_CHECK(cudaMemcpy(b->d_blocks, h_blocks,
        (size_t)b->n_blocks * sizeof(IRBlockInfoGPU), cudaMemcpyHostToDevice));
    free(h_blocks);

    /* IR instructions */
    CUDA_CHECK(cudaMalloc(&b->d_insns,
        (size_t)b->n_insns * sizeof(IRInsn)));
    CUDA_CHECK(cudaMemcpy(b->d_insns, h_insns,
        (size_t)b->n_insns * sizeof(IRInsn), cudaMemcpyHostToDevice));
    free(h_insns);

    /* ── Build O(1) block hash table ── */
    {
        BlockHashSlot *h_hash = (BlockHashSlot *)malloc(
            GPU_BLOCK_HASH_SIZE * sizeof(BlockHashSlot));
        if (!h_hash) { free(b); return NULL; }
        for (int i = 0; i < GPU_BLOCK_HASH_SIZE; i++)
            h_hash[i].key = GPU_BLOCK_HASH_EMPTY;

        /* Insert all blocks; Knuth multiplicative hash with linear probing */
        IRBlockInfoGPU *h_blk_tmp = (IRBlockInfoGPU *)malloc(
            (size_t)b->n_blocks * sizeof(IRBlockInfoGPU));
        if (!h_blk_tmp) { free(h_hash); free(b); return NULL; }
        CUDA_CHECK(cudaMemcpy(h_blk_tmp, b->d_blocks,
            (size_t)b->n_blocks * sizeof(IRBlockInfoGPU),
            cudaMemcpyDeviceToHost));

        for (int i = 0; i < b->n_blocks; i++) {
            uint32_t key = h_blk_tmp[i].key;
            uint32_t h   = (key * 2654435761u) >> (32 - GPU_BLOCK_HASH_BITS);
            while (h_hash[h].key != GPU_BLOCK_HASH_EMPTY)
                h = (h + 1) & (GPU_BLOCK_HASH_SIZE - 1);
            h_hash[h].key = key;
            h_hash[h].idx = i;
        }
        free(h_blk_tmp);

        CUDA_CHECK(cudaMalloc(&b->d_hash,
            GPU_BLOCK_HASH_SIZE * sizeof(BlockHashSlot)));
        CUDA_CHECK(cudaMemcpy(b->d_hash, h_hash,
            GPU_BLOCK_HASH_SIZE * sizeof(BlockHashSlot),
            cudaMemcpyHostToDevice));
        free(h_hash);
        fprintf(stderr, "GPU: hash table: %d slots, %.1f%% load (%.1f MB)\n",
                GPU_BLOCK_HASH_SIZE,
                100.0 * b->n_blocks / GPU_BLOCK_HASH_SIZE,
                (double)(GPU_BLOCK_HASH_SIZE * sizeof(BlockHashSlot)) / (1024*1024));
    }

    /* ── Actions + sorted-dispatch buffers ── */
    CUDA_CHECK(cudaMalloc(&b->d_actions, (size_t)n_states));
    CUDA_CHECK(cudaMemset(b->d_actions, 0, (size_t)n_states));

    CUDA_CHECK(cudaMalloc(&b->d_sorted,     (size_t)n_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b->d_sorted_buf, (size_t)n_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b->d_keys,       (size_t)n_states * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b->d_keys_buf,   (size_t)n_states * sizeof(uint32_t)));

    /* Query CUB temp storage size */
    b->cub_tmp_size = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, b->cub_tmp_size,
        (uint32_t *)nullptr, (uint32_t *)nullptr,
        (int *)nullptr,      (int *)nullptr,
        n_states);
    CUDA_CHECK(cudaMalloc(&b->d_cub_tmp, b->cub_tmp_size));

    /* Profiling buffer starts NULL (off) */
    b->d_profile = nullptr;
    b->h_profile = nullptr;

    /* Upload JIT trace hash to __constant__ memory (stub = all misses) */
    trace_hash_upload();

    return b;
}

void gpu_batch_run_frames(GPUBatch *b, int n_frames)
{
    int threads = 128;
    int blks    = (b->n_states + threads - 1) / threads;

    NVTX_PUSH("sort_dispatch");
    gpu_iota<<<blks, threads>>>(b->d_sorted, b->n_states);
    gpu_compute_block_keys<<<blks, threads>>>(
        b->d_states, b->d_keys, b->n_states);

    cub::DeviceRadixSort::SortPairs(
        b->d_cub_tmp, b->cub_tmp_size,
        b->d_keys,    b->d_keys_buf,
        b->d_sorted,  b->d_sorted_buf,
        b->n_states);
    NVTX_POP();

    NVTX_PUSH("run_frames_kernel");
    gpu_run_frames_sorted_kernel<<<blks, threads>>>(
        b->d_states,
        b->d_vrams,
        b->d_wrams,
        b->d_ext_rams,
        b->d_rom,
        b->rom_size,
        b->ext_ram_size,
        b->d_blocks,
        b->n_blocks,
        b->d_insns,
        b->d_actions,
        b->d_sorted_buf,
        b->d_hash,
        b->d_profile,    /* NULL when not profiling */
        b->n_states,
        n_frames);

    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();
}

void gpu_batch_set_actions(GPUBatch *b, const uint8_t *host_actions)
{
    CUDA_CHECK(cudaMemcpy(b->d_actions, host_actions,
        (size_t)b->n_states, cudaMemcpyHostToDevice));
}

void gpu_batch_get_actions(const GPUBatch *b, uint8_t *host_actions)
{
    CUDA_CHECK(cudaMemcpy(host_actions, b->d_actions,
        (size_t)b->n_states, cudaMemcpyDeviceToHost));
}

void gpu_batch_sync_to_host(
    const GPUBatch *b,
    GB             *host_states,
    uint8_t        *host_ext_rams)
{
    GBStateGPU *h_states = (GBStateGPU *)malloc(
        (size_t)b->n_states * sizeof(GBStateGPU));
    uint8_t    *h_vrams  = (uint8_t *)malloc(
        (size_t)b->n_states * GPU_VRAM_SIZE);
    uint8_t    *h_wrams  = (uint8_t *)malloc(
        (size_t)b->n_states * GPU_WRAM_SIZE);
    if (!h_states || !h_vrams || !h_wrams) {
        free(h_states); free(h_vrams); free(h_wrams); return;
    }

    CUDA_CHECK(cudaMemcpy(h_states, b->d_states,
        (size_t)b->n_states * sizeof(GBStateGPU), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vrams, b->d_vrams,
        (size_t)b->n_states * GPU_VRAM_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wrams, b->d_wrams,
        (size_t)b->n_states * GPU_WRAM_SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < b->n_states; i++) {
        gpu_state_to_gb(&h_states[i], &host_states[i]);
        memcpy(host_states[i].mem.vram,
               h_vrams + (size_t)i * GPU_VRAM_SIZE, GPU_VRAM_SIZE);
        memcpy(host_states[i].mem.wram,
               h_wrams + (size_t)i * GPU_WRAM_SIZE, GPU_WRAM_SIZE);
    }
    free(h_states);
    free(h_vrams);
    free(h_wrams);

    if (b->ext_ram_size > 0 && host_ext_rams) {
        CUDA_CHECK(cudaMemcpy(host_ext_rams, b->d_ext_rams,
            (size_t)b->n_states * b->ext_ram_size, cudaMemcpyDeviceToHost));
    }
}

void gpu_batch_print_stats(const GPUBatch *b)
{
    /* Copy first state back for display */
    GBStateGPU g;
    CUDA_CHECK(cudaMemcpy(&g, b->d_states, sizeof(GBStateGPU),
                          cudaMemcpyDeviceToHost));
    fprintf(stderr,
        "GPU Batch: %d states  frame=%u  PC=%04X  cycles=%llu\n"
        "  IR: %d blocks  %d insns\n"
        "  VRAM: %.1f MB  per state: %zu KB\n"
        "  Fallback: %u calls  %llu cycles\n",
        b->n_states,
        g.ppu_frame, g.PC,
        (unsigned long long)g.cycles,
        b->n_blocks, b->n_insns,
        (double)((size_t)b->n_states * sizeof(GBStateGPU)) / (1024.0*1024.0),
        sizeof(GBStateGPU) / 1024,
        g.fallback_count, (unsigned long long)g.fallback_cycles);
}

void gpu_batch_free(GPUBatch *b)
{
    if (!b) return;
    if (b->d_states)     cudaFree(b->d_states);
    if (b->d_vrams)      cudaFree(b->d_vrams);
    if (b->d_wrams)      cudaFree(b->d_wrams);
    if (b->d_ext_rams)   cudaFree(b->d_ext_rams);
    if (b->d_rom)        cudaFree((void *)b->d_rom);
    if (b->d_blocks)     cudaFree(b->d_blocks);
    if (b->d_insns)      cudaFree(b->d_insns);
    if (b->d_hash)       cudaFree(b->d_hash);
    if (b->d_actions)    cudaFree(b->d_actions);
    if (b->d_sorted)     cudaFree(b->d_sorted);
    if (b->d_sorted_buf) cudaFree(b->d_sorted_buf);
    if (b->d_keys)       cudaFree(b->d_keys);
    if (b->d_keys_buf)   cudaFree(b->d_keys_buf);
    if (b->d_cub_tmp)    cudaFree(b->d_cub_tmp);
    if (b->d_profile)    cudaFree(b->d_profile);
    free(b->h_profile);
    free(b);
}

/* ============================================================
 * Block-hit profiling API
 * ============================================================ */

void gpu_batch_start_profiling(GPUBatch *b)
{
    if (!b || b->d_profile) return;   /* already profiling */
    CUDA_CHECK(cudaMalloc(&b->d_profile,
                          (size_t)b->n_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(b->d_profile, 0,
                          (size_t)b->n_blocks * sizeof(uint32_t)));
    fprintf(stderr, "Profiling: tracking %d blocks.\n", b->n_blocks);
}

void gpu_batch_stop_profiling(GPUBatch *b)
{
    if (!b || !b->d_profile) return;

    free(b->h_profile);
    b->h_profile = (uint32_t *)malloc(
        (size_t)b->n_blocks * sizeof(uint32_t));
    if (!b->h_profile) { fprintf(stderr, "OOM in stop_profiling\n"); return; }

    CUDA_CHECK(cudaMemcpy(b->h_profile, b->d_profile,
                          (size_t)b->n_blocks * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(b->d_profile);
    b->d_profile = nullptr;

    /* Count non-zero blocks for a quick summary */
    uint64_t total = 0;
    int hot = 0;
    for (int i = 0; i < b->n_blocks; i++) {
        total += b->h_profile[i];
        if (b->h_profile[i] > 0) hot++;
    }
    fprintf(stderr, "Profiling: %d / %d blocks hit, %llu total executions.\n",
            hot, b->n_blocks, (unsigned long long)total);
}

void gpu_batch_dump_profile(const GPUBatch *b, const char *path)
{
    if (!b || !b->h_profile) {
        fprintf(stderr, "gpu_batch_dump_profile: no profile data "
                        "(call gpu_batch_stop_profiling first)\n");
        return;
    }
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write profile: %s\n", path); return; }
    uint32_t n = (uint32_t)b->n_blocks;
    fwrite(&n, sizeof(uint32_t), 1, f);
    fwrite(b->h_profile, sizeof(uint32_t), (size_t)n, f);
    fclose(f);
    fprintf(stderr, "Profile written: %s  (%u blocks)\n", path, n);
}

/* ============================================================
 * RL episode kernel + host API
 * ============================================================ */

#include "rl.h"

/* Device-side action map (mirrors RL_ACTION_MAP in rl_gpu.h) */
__constant__ static uint8_t c_rl_action_map[RL_N_ACTIONS] = {
    0x00,  /* 0: nothing */
    0x40,  /* 1: up      */
    0x80,  /* 2: down    */
    0x20,  /* 3: left    */
    0x10,  /* 4: right   */
    0x01,  /* 5: A       */
    0x02,  /* 6: B       */
    0x08,  /* 7: start   */
    0x04,  /* 8: select  */
};

/* Pokémon Red WRAM offsets for player position:
 *   wCurMap  0xD35E  → wram[0x135E]
 *   wYCoord  0xD361  → wram[0x1361]
 *   wXCoord  0xD362  → wram[0x1362] */
#define WRAM_CUR_MAP  0x135Eu
#define WRAM_Y_COORD  0x1361u
#define WRAM_X_COORD  0x1362u

/* ── ext_ram broadcast kernel ──────────────────────────────── */
/* Copies a single ext_ram template to all N agent slots.
 * Uses a grid-stride loop so it works for any N and any ext_ram_size.
 * Assumes ext_ram_size is a power of two (always true for GB/GBC). */
__global__ void rl_broadcast_ext_ram_kernel(
    uint8_t       *dst,
    const uint8_t *src,
    uint64_t       n_states,
    uint32_t       ext_ram_size)
{
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t total  = n_states * (uint64_t)ext_ram_size;
    uint32_t mask   = ext_ram_size - 1u;  /* valid because ext_ram_size is pow2 */
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride)
        dst[i] = src[(uint32_t)i & mask];
}

/* ── Episode kernel ────────────────────────────────────────── */

__global__ void rl_episode_kernel(
    GBStateGPU          *states,
    uint8_t             *d_vrams,
    uint8_t             *d_wrams,
    uint8_t             *ext_rams,
    const uint8_t       *rom,
    uint32_t             rom_size,
    uint32_t             ext_ram_size,
    const IRBlockInfoGPU *blocks,
    const IRInsn        *insns,
    const BlockHashSlot *hash,
    uint32_t            *d_profile,   /* NULL = off */
    const uint8_t       *act_seqs,    /* [n_states × action_steps] action indices */
    uint32_t            *tile_sets,   /* [n_states × RL_TILE_HASH_SIZE] */
    uint32_t            *tile_counts, /* [n_states] */
    const uint8_t       *d_alive,     /* NULL = all alive; 0 = skip this thread */
    int                  n_states,
    int                  action_steps,  /* number of action decisions */
    int                  action_repeat) /* GB frames each action is held */
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_states) return;
    /* Successive Halving: skip culled agents entirely (zero compute). */
    if (d_alive && d_alive[tid] == 0) {
        tile_counts[tid] = 0;
        return;
    }

    GBStateGPU *s       = &states[tid];
    uint8_t    *vram    = d_vrams + (uint64_t)tid * GPU_VRAM_SIZE;
    uint8_t    *wram    = d_wrams + (uint64_t)tid * GPU_WRAM_SIZE;
    uint8_t    *ext_ram = (ext_ram_size > 0)
                          ? (ext_rams + (uint64_t)tid * ext_ram_size)
                          : nullptr;

    /* Per-state visited-tile hash set */
    uint32_t      *tset = tile_sets + (uint64_t)tid * RL_TILE_HASH_SIZE;
    const uint8_t *aseq = act_seqs  + (uint64_t)tid * action_steps;

    /* Clear tile hash set */
    for (int i = 0; i < RL_TILE_HASH_SIZE; i++)
        tset[i] = RL_TILE_HASH_EMPTY;
    uint32_t n_unique = 0;

    uint32_t start_frame = s->ppu_frame;
    uint8_t  prev_map    = wram[WRAM_CUR_MAP];

    /* Pre-insert the starting map's sentinel so returning to it gives no reward.
     * Only NEW maps visited for the first time during the episode score +1. */
    {
        uint32_t start_id = ((uint32_t)prev_map << 16) | RL_MAP_SENTINEL;
        uint32_t sh = (start_id * 2654435761u) >> (32 - RL_TILE_HASH_BITS);
        for (int probe = 0; probe < RL_TILE_HASH_SIZE; probe++) {
            uint32_t k = tset[sh];
            if (k == RL_TILE_HASH_EMPTY) { tset[sh] = start_id; break; }
            sh = (sh + 1) & (RL_TILE_HASH_SIZE - 1);
        }
        /* n_unique NOT incremented — start map is free */
    }

    for (int step = 0; step < action_steps; step++) {
        /* Apply action — held for action_repeat GB frames */
        uint8_t act_idx = aseq[step];
        s->joypad_action = (act_idx < RL_N_ACTIONS) ? c_rl_action_map[act_idx] : 0u;

        /* Advance action_repeat PPU frames */
        uint32_t target = start_frame + (uint32_t)((step + 1) * action_repeat);
        while (s->ppu_frame < target) {
            gpu_ir_step(s, vram, wram, ext_ram, rom, rom_size, ext_ram_size,
                        blocks, insns, hash, d_profile);
        }

        /* Sample player tile position from WRAM */
        uint8_t  cur_map = wram[WRAM_CUR_MAP];
        uint8_t  py      = wram[WRAM_Y_COORD];
        uint8_t  px      = wram[WRAM_X_COORD];

        /* Entering a new map: insert a virtual sentinel tile (map, 0xFF, 0xFF).
         * Counts as exactly +1 in the same hash set — can't double-count,
         * and matches what playback tracks, so scores are comparable. */
        if (cur_map != prev_map) {
            uint32_t map_id = ((uint32_t)cur_map << 16) | RL_MAP_SENTINEL;
            uint32_t hm = (map_id * 2654435761u) >> (32 - RL_TILE_HASH_BITS);
            for (int probe = 0; probe < RL_TILE_HASH_SIZE; probe++) {
                uint32_t k = tset[hm];
                if (k == map_id)             break;
                if (k == RL_TILE_HASH_EMPTY) { tset[hm] = map_id; n_unique++; break; }
                hm = (hm + 1) & (RL_TILE_HASH_SIZE - 1);
            }
            prev_map = cur_map;
        }

        uint32_t tile_id = ((uint32_t)cur_map << 16)
                         | ((uint32_t)py      <<  8)
                         |  (uint32_t)px;

        /* Insert into open-addressing hash set */
        uint32_t h = (tile_id * 2654435761u) >> (32 - RL_TILE_HASH_BITS);
        for (int probe = 0; probe < RL_TILE_HASH_SIZE; probe++) {
            uint32_t k = tset[h];
            if (k == tile_id)            break;  /* already visited */
            if (k == RL_TILE_HASH_EMPTY) {
                tset[h] = tile_id;
                n_unique++;
                break;
            }
            h = (h + 1) & (RL_TILE_HASH_SIZE - 1);
        }
    }

    tile_counts[tid] = n_unique;
}

/* ── Host API ──────────────────────────────────────────────── */

RLBatch *rl_batch_create(GPUBatch *batch, int ep_frames, int action_repeat)
{
    if (!batch || ep_frames <= 0 || action_repeat <= 0) return NULL;

    RLBatch *r = (RLBatch *)calloc(1, sizeof(*r));
    if (!r) return NULL;

    r->batch        = batch;
    r->n_states     = batch->n_states;
    r->ep_frames    = ep_frames;
    r->action_repeat = action_repeat;
    r->action_steps  = ep_frames / action_repeat;
    if (r->action_steps < 1) r->action_steps = 1;

    int    N   = r->n_states;
    int    AS  = r->action_steps;
    size_t ers = (size_t)batch->ext_ram_size;

    /* Action sequences — one byte per action step (not per frame) */
    CUDA_CHECK(cudaMalloc(&r->d_act_seqs, (size_t)N * AS));

    /* Tile hash sets */
    CUDA_CHECK(cudaMalloc(&r->d_tile_sets,
                          (size_t)N * RL_TILE_HASH_SIZE * sizeof(uint32_t)));

    /* Tile counts */
    CUDA_CHECK(cudaMalloc(&r->d_tile_counts, (size_t)N * sizeof(uint32_t)));

    /* Snapshot current state for episode resets.
     * All agents are clones of the same post-intro state, so ONE ext_ram template
     * is sufficient — a broadcast kernel fans it out to all N slots each episode. */
    CUDA_CHECK(cudaMalloc(&r->d_init_states, (size_t)N * sizeof(GBStateGPU)));
    CUDA_CHECK(cudaMemcpy(r->d_init_states, batch->d_states,
                          (size_t)N * sizeof(GBStateGPU),
                          cudaMemcpyDeviceToDevice));

    if (ers > 0) {
        /* Single template instead of N full copies */
        CUDA_CHECK(cudaMalloc(&r->d_init_ext_ram, ers));
        CUDA_CHECK(cudaMemcpy(r->d_init_ext_ram, batch->d_ext_rams,
                              ers, cudaMemcpyDeviceToDevice));
    }

    /* Single 8KB VRAM template (all agents start from the same VRAM) */
    CUDA_CHECK(cudaMalloc(&r->d_init_vram, GPU_VRAM_SIZE));
    CUDA_CHECK(cudaMemcpy(r->d_init_vram, batch->d_vrams,
                          GPU_VRAM_SIZE, cudaMemcpyDeviceToDevice));

    /* Single 8KB WRAM template */
    CUDA_CHECK(cudaMalloc(&r->d_init_wram, GPU_WRAM_SIZE));
    CUDA_CHECK(cudaMemcpy(r->d_init_wram, batch->d_wrams,
                          GPU_WRAM_SIZE, cudaMemcpyDeviceToDevice));

    /* Host mirrors */
    r->h_tile_counts = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
    r->h_act_seqs    = (uint8_t  *)malloc((size_t)N * AS);
    if (!r->h_tile_counts || !r->h_act_seqs) {
        rl_batch_free(r);
        return NULL;
    }

    /* Default: all actions = 0 (nothing) */
    memset(r->h_act_seqs, 0, (size_t)N * AS);
    CUDA_CHECK(cudaMemcpy(r->d_act_seqs, r->h_act_seqs,
                          (size_t)N * AS,
                          cudaMemcpyHostToDevice));

    /* Report extra VRAM used by RLBatch (on top of GPUBatch) */
    size_t extra_mb = ((size_t)N * sizeof(GBStateGPU)   /* d_init_states */
                     + ers                               /* d_init_ext_ram: 1 template */
                     + (size_t)N * RL_TILE_HASH_SIZE * sizeof(uint32_t)  /* d_tile_sets */
                     + (size_t)N * AS) / (1024 * 1024); /* d_act_seqs */
    fprintf(stderr,
            "RL batch: %d states  ep_frames=%d  action_steps=%d  repeat=%d"
            "  extra VRAM: ~%zu MB  (hash=%d slots/state)\n",
            N, ep_frames, AS, action_repeat, extra_mb, RL_TILE_HASH_SIZE);

    return r;
}

void rl_batch_free(RLBatch *r)
{
    if (!r) return;
    if (r->d_act_seqs)     cudaFree(r->d_act_seqs);
    if (r->d_tile_sets)    cudaFree(r->d_tile_sets);
    if (r->d_tile_counts)  cudaFree(r->d_tile_counts);
    if (r->d_init_states)  cudaFree(r->d_init_states);
    if (r->d_init_ext_ram) cudaFree(r->d_init_ext_ram);
    if (r->d_init_vram)    cudaFree(r->d_init_vram);
    if (r->d_init_wram)    cudaFree(r->d_init_wram);
    if (r->d_alive)        cudaFree(r->d_alive);
    free(r->h_tile_counts);
    free(r->h_act_seqs);
    free(r);
}

void rl_episode_set_alive(RLBatch *r, const uint8_t *host_alive)
{
    if (!r) return;
    if (!host_alive) {
        /* NULL → all alive: free device mask */
        if (r->d_alive) { cudaFree(r->d_alive); r->d_alive = nullptr; }
        return;
    }
    if (!r->d_alive)
        CUDA_CHECK(cudaMalloc(&r->d_alive, (size_t)r->n_states));
    CUDA_CHECK(cudaMemcpy(r->d_alive, host_alive,
                          (size_t)r->n_states, cudaMemcpyHostToDevice));
}

void rl_batch_set_seqs(RLBatch *r, const uint8_t *h_seqs)
{
    size_t sz = (size_t)r->n_states * r->action_steps;
    memcpy(r->h_act_seqs, h_seqs, sz);
    CUDA_CHECK(cudaMemcpy(r->d_act_seqs, h_seqs, sz, cudaMemcpyHostToDevice));
}

void rl_episode_run(RLBatch *r)
{
    GPUBatch *b = r->batch;
    int N       = r->n_states;
    size_t ers  = (size_t)b->ext_ram_size;

    NVTX_PUSH("rl_episode_reset");
    CUDA_CHECK(cudaMemcpy(b->d_states, r->d_init_states,
                          (size_t)N * sizeof(GBStateGPU),
                          cudaMemcpyDeviceToDevice));
    if (ers > 0 && r->d_init_ext_ram)
        rl_broadcast_ext_ram_kernel<<<32768, 256>>>(
            b->d_ext_rams, r->d_init_ext_ram, (uint64_t)N, (uint32_t)ers);
    rl_broadcast_ext_ram_kernel<<<32768, 256>>>(
        b->d_vrams, r->d_init_vram, (uint64_t)N, (uint32_t)GPU_VRAM_SIZE);
    rl_broadcast_ext_ram_kernel<<<32768, 256>>>(
        b->d_wrams, r->d_init_wram, (uint64_t)N, (uint32_t)GPU_WRAM_SIZE);
    NVTX_POP();

    /* Prefer L1 cache over shared memory — GBStateGPU doesn't fit in smem. */
    CUDA_CHECK(cudaFuncSetCacheConfig(rl_episode_kernel, cudaFuncCachePreferL1));

    int threads = 128;
    int blocks  = (N + threads - 1) / threads;

    NVTX_PUSH("rl_episode_kernel");
    rl_episode_kernel<<<blocks, threads>>>(
        b->d_states,
        b->d_vrams,
        b->d_wrams,
        b->d_ext_rams,
        b->d_rom,
        b->rom_size,
        b->ext_ram_size,
        b->d_blocks,
        b->d_insns,
        b->d_hash,
        b->d_profile,
        r->d_act_seqs,
        r->d_tile_sets,
        r->d_tile_counts,
        r->d_alive,
        N,
        r->action_steps,
        r->action_repeat);
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();
}

void rl_episode_get_counts(RLBatch *r)
{
    CUDA_CHECK(cudaMemcpy(r->h_tile_counts, r->d_tile_counts,
                          (size_t)r->n_states * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}
