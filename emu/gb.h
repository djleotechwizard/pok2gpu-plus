#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* =========================================================
 * GB — master header: all type definitions and declarations
 * ========================================================= */

typedef struct GB GB;

/* ── CPU ─────────────────────────────────────────────────── */

#define FLAG_Z 0x80
#define FLAG_N 0x40
#define FLAG_H 0x20
#define FLAG_C 0x10

typedef struct {
    uint8_t  A, F;
    uint8_t  B, C;
    uint8_t  D, E;
    uint8_t  H, L;
    uint16_t SP;
    uint16_t PC;

    bool     IME;           /* interrupt master enable */
    bool     ime_pending;   /* EI takes effect after next instruction */
    bool     halted;
    bool     halt_bug;      /* HALT bug: PC not incremented on next fetch */

    uint64_t cycles;        /* total T-cycles elapsed */
} CPU;

static inline uint16_t cpu_AF(const CPU *c) { return ((uint16_t)c->A << 8) | (c->F & 0xF0); }
static inline uint16_t cpu_BC(const CPU *c) { return ((uint16_t)c->B << 8) | c->C; }
static inline uint16_t cpu_DE(const CPU *c) { return ((uint16_t)c->D << 8) | c->E; }
static inline uint16_t cpu_HL(const CPU *c) { return ((uint16_t)c->H << 8) | c->L; }
static inline void cpu_set_AF(CPU *c, uint16_t v) { c->A = v >> 8; c->F = (uint8_t)(v & 0xF0); }
static inline void cpu_set_BC(CPU *c, uint16_t v) { c->B = v >> 8; c->C = v & 0xFF; }
static inline void cpu_set_DE(CPU *c, uint16_t v) { c->D = v >> 8; c->E = v & 0xFF; }
static inline void cpu_set_HL(CPU *c, uint16_t v) { c->H = v >> 8; c->L = v & 0xFF; }

#define GET_Z(c)     (((c)->F & FLAG_Z) != 0)
#define GET_N(c)     (((c)->F & FLAG_N) != 0)
#define GET_H(c)     (((c)->F & FLAG_H) != 0)
#define GET_C(c)     (((c)->F & FLAG_C) != 0)
#define SET_Z(c,v)   do { if(v) (c)->F |= FLAG_Z; else (c)->F &= ~FLAG_Z; } while(0)
#define SET_N(c,v)   do { if(v) (c)->F |= FLAG_N; else (c)->F &= ~FLAG_N; } while(0)
#define SET_H(c,v)   do { if(v) (c)->F |= FLAG_H; else (c)->F &= ~FLAG_H; } while(0)
#define SET_C(c,v)   do { if(v) (c)->F |= FLAG_C; else (c)->F &= ~FLAG_C; } while(0)

/* ── Memory ──────────────────────────────────────────────── */

typedef enum { MBC_NONE=0, MBC_1=1, MBC_3=3, MBC_5=5 } MBCType;

typedef struct {
    uint8_t *rom;
    uint32_t rom_size;
    uint8_t *ext_ram;
    uint32_t ext_ram_size;

    MBCType  mbc_type;
    uint8_t  rom_bank;      /* current upper ROM bank (0x4000–0x7FFF) */
    uint8_t  ram_bank;
    bool     ram_enabled;
    int      mbc1_mode;     /* 0=ROM banking, 1=RAM banking */

    uint8_t  wram[0x2000];
    uint8_t  hram[0x7F];
    uint8_t  vram[0x2000];
    uint8_t  oam[0xA0];
    uint8_t  io[0x80];      /* 0xFF00–0xFF7F */
    uint8_t  IE;            /* 0xFFFF */

    /* Joypad state: bit0=A bit1=B bit2=sel bit3=start
     *               bit4=right bit5=left bit6=up bit7=down (active high) */
    uint8_t  joypad_action;

    /* OAM DMA */
    bool     dma_active;
    uint16_t dma_src;
    int      dma_index;
} Memory;

/* ── PPU ─────────────────────────────────────────────────── */

typedef enum {
    PPU_HBLANK = 0,
    PPU_VBLANK = 1,
    PPU_OAM    = 2,
    PPU_DRAW   = 3,
} PPUMode;

typedef struct {
    int      scanline;  /* LY: 0–153 */
    int      dot;       /* T-cycles within current line (0–455) */
    PPUMode  mode;
    uint32_t frame;
    uint8_t  framebuffer[160 * 144];
} PPU;

/* ── Timer ───────────────────────────────────────────────── */

typedef struct {
    uint16_t div_internal;  /* internal 16-bit counter; DIV = upper byte */
    uint8_t  tima;
    uint8_t  tma;
    uint8_t  tac;
    bool     tima_overflow;
    int      overflow_delay;
} Timer;

/* ── Trace ───────────────────────────────────────────────── */

#define TRACE_MAX_MEM_WRITES 16

typedef struct {
    uint64_t cycle;
    uint16_t pc;
    uint8_t  opcode[4];
    int      opcode_len;
    uint8_t  A, F, B, C, D, E, H, L;
    uint16_t SP;
    uint16_t PC_after;
    struct { uint16_t addr; uint8_t value; } mem_writes[TRACE_MAX_MEM_WRITES];
    int      mem_write_count;
    bool     interrupt_fired;
    uint8_t  interrupt_vector;
} TraceEntry;

typedef struct {
    FILE    *file;
    bool     enabled;
    uint64_t entry_count;
} Tracer;

/* ── Main GB struct ──────────────────────────────────────── */

struct GB {
    CPU    cpu;
    Memory mem;
    PPU    ppu;
    Timer  timer;
    Tracer tracer;

    TraceEntry pending_trace;
};

/* ── Function declarations ───────────────────────────────── */

void cpu_init(CPU *cpu);
int  cpu_step(GB *gb);

void    memory_init(Memory *mem);
int     memory_load_rom(Memory *mem, const char *path);
uint8_t memory_read(GB *gb, uint16_t addr);
void    memory_write(GB *gb, uint16_t addr, uint8_t value);
void    memory_dma_tick(GB *gb, int cycles);

void ppu_init(PPU *ppu);
void ppu_step(GB *gb, int cycles);
void ppu_save_ppm(const PPU *ppu, const char *path);

void timer_init(Timer *t);
void timer_step(GB *gb, int cycles);

void tracer_init(Tracer *t, const char *path);
void tracer_close(Tracer *t);
void tracer_emit(Tracer *t, const TraceEntry *e);

void gb_init(GB *gb);
int  gb_load_rom(GB *gb, const char *path);
void gb_step(GB *gb);
void gb_run_frames(GB *gb, int n);
void gb_enable_trace(GB *gb, const char *path);
void gb_disable_trace(GB *gb);
