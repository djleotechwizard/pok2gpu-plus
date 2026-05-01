#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gb.h"

#define IO_JOYP  0x00
#define IO_SC    0x02
#define IO_DIV   0x04
#define IO_TIMA  0x05
#define IO_TMA   0x06
#define IO_TAC   0x07
#define IO_IF    0x0F
#define IO_LCDC  0x40
#define IO_STAT  0x41
#define IO_LY    0x44
#define IO_LYC   0x45
#define IO_DMA   0x46
#define IO_BGP   0x47
#define IO_OBP0  0x48
#define IO_OBP1  0x49

void memory_init(Memory *mem) {
    memset(mem, 0, sizeof(*mem));
    mem->rom_bank    = 1;
    mem->ram_bank    = 0;
    mem->ram_enabled = false;
    mem->mbc1_mode   = 0;

    mem->io[IO_JOYP] = 0xCF;
    mem->io[IO_SC]   = 0x7E;
    mem->io[IO_TAC]  = 0xF8;
    mem->io[IO_IF]   = 0xE1;
    mem->io[IO_LCDC] = 0x91;
    mem->io[IO_STAT] = 0x85;
    mem->io[IO_BGP]  = 0xFC;
    mem->io[IO_OBP0] = 0xFF;
    mem->io[IO_OBP1] = 0xFF;
    mem->IE          = 0x00;
}

int memory_load_rom(Memory *mem, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open ROM: %s\n", path); return -1; }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    mem->rom = (uint8_t *)malloc((size_t)size);
    if (!mem->rom) { fclose(f); return -1; }
    if (fread(mem->rom, 1, (size_t)size, f) != (size_t)size) {
        free(mem->rom); mem->rom = NULL; fclose(f); return -1;
    }
    fclose(f);
    mem->rom_size = (uint32_t)size;

    uint8_t cart_type = mem->rom[0x147];
    switch (cart_type) {
        case 0x00:                        mem->mbc_type = MBC_NONE; break;
        case 0x01: case 0x02: case 0x03: mem->mbc_type = MBC_1;    break;
        case 0x10: case 0x11:
        case 0x12: case 0x13:            mem->mbc_type = MBC_3;    break;
        case 0x19: case 0x1A: case 0x1B:
        case 0x1C: case 0x1D: case 0x1E: mem->mbc_type = MBC_5;    break;
        default:
            fprintf(stderr, "Warning: unknown cart type 0x%02X, assuming MBC1\n", cart_type);
            mem->mbc_type = MBC_1;
    }

    static const uint32_t ram_sizes[] = { 0, 0x800, 0x2000, 0x8000, 0x20000, 0x10000 };
    uint8_t rs = mem->rom[0x149];
    mem->ext_ram_size = (rs < 6) ? ram_sizes[rs] : 0;
    if (mem->ext_ram_size > 0) {
        mem->ext_ram = (uint8_t *)malloc(mem->ext_ram_size);
        /* 0xFF = unwritten SRAM default — avoids Pokémon Red's checksum false-positive */
        if (mem->ext_ram) memset(mem->ext_ram, 0xFF, mem->ext_ram_size);
    }

    fprintf(stderr, "ROM loaded: %u bytes, cart_type=0x%02X, MBC=%d, ext_ram=%u bytes\n",
            mem->rom_size, cart_type, (int)mem->mbc_type, mem->ext_ram_size);
    return 0;
}

static uint8_t *rom_bank_ptr(const Memory *mem, int bank) {
    uint32_t offset = (uint32_t)(bank & 0xFF) * 0x4000;
    if (offset >= mem->rom_size) offset = offset % mem->rom_size;
    return mem->rom + offset;
}

static uint8_t *ext_ram_ptr(const Memory *mem, int bank) {
    if (!mem->ext_ram || mem->ext_ram_size == 0) return NULL;
    uint32_t offset = (uint32_t)(bank & 0x0F) * 0x2000;
    if (offset >= mem->ext_ram_size) offset = offset % mem->ext_ram_size;
    return mem->ext_ram + offset;
}

uint8_t memory_read(GB *gb, uint16_t addr) {
    Memory *mem = &gb->mem;

    if (addr < 0x4000) return (addr < mem->rom_size) ? mem->rom[addr] : 0xFF;
    if (addr < 0x8000) {
        int bank = mem->rom_bank ? mem->rom_bank : 1;
        return rom_bank_ptr(mem, bank)[addr - 0x4000];
    }
    if (addr < 0xA000) return mem->vram[addr - 0x8000];
    if (addr < 0xC000) {
        if (!mem->ram_enabled || !mem->ext_ram) return 0xFF;
        uint8_t *p = ext_ram_ptr(mem, mem->ram_bank);
        return p ? p[addr - 0xA000] : 0xFF;
    }
    if (addr < 0xE000) return mem->wram[addr - 0xC000];
    if (addr < 0xFE00) return mem->wram[(addr - 0xE000) & 0x1FFF];
    if (addr < 0xFEA0) return mem->oam[addr - 0xFE00];
    if (addr < 0xFF00) return 0xFF;
    if (addr < 0xFF80) {
        uint8_t reg = (uint8_t)(addr - 0xFF00);
        if (reg == IO_DIV)  return (uint8_t)(gb->timer.div_internal >> 8);
        if (reg == IO_TIMA) return gb->timer.tima;
        if (reg == IO_TMA)  return gb->timer.tma;
        if (reg == IO_TAC)  return gb->timer.tac | 0xF8;
        if (reg == IO_LY)   return (uint8_t)gb->ppu.scanline;
        if (reg == IO_STAT) {
            uint8_t stat = mem->io[IO_STAT] & 0xF8;
            stat |= (uint8_t)gb->ppu.mode & 0x03;
            if (gb->ppu.scanline == (int)mem->io[IO_LYC])
                stat |= 0x04;
            return stat;
        }
        if (reg == IO_JOYP) {
            uint8_t sel = mem->io[IO_JOYP];
            uint8_t lo  = 0x0Fu;
            if (!(sel & 0x20u)) lo &= ~(mem->joypad_action & 0x0Fu);
            if (!(sel & 0x10u)) lo &= ~((mem->joypad_action >> 4) & 0x0Fu);
            return (sel & 0x30u) | 0xC0u | lo;
        }
        return mem->io[reg];
    }
    if (addr < 0xFFFF) return mem->hram[addr - 0xFF80];
    return mem->IE;
}

void memory_write(GB *gb, uint16_t addr, uint8_t value) {
    Memory *mem = &gb->mem;

    if (gb->tracer.enabled) {
        TraceEntry *te = &gb->pending_trace;
        if (te->mem_write_count < TRACE_MAX_MEM_WRITES) {
            te->mem_writes[te->mem_write_count].addr  = addr;
            te->mem_writes[te->mem_write_count].value = value;
            te->mem_write_count++;
        }
    }

    if (addr < 0x2000) { mem->ram_enabled = (value & 0x0F) == 0x0A; return; }
    if (addr < 0x4000) {
        if (mem->mbc_type == MBC_1) {
            uint8_t b = value & 0x1F; if (!b) b = 1;
            mem->rom_bank = (mem->rom_bank & 0x60) | b;
        } else if (mem->mbc_type == MBC_3) {
            uint8_t b = value & 0x7F; if (!b) b = 1;
            mem->rom_bank = b;
        } else if (mem->mbc_type == MBC_5) {
            mem->rom_bank = value;
        }
        return;
    }
    if (addr < 0x6000) {
        if (mem->mbc_type == MBC_1) {
            if (mem->mbc1_mode == 0)
                mem->rom_bank = (mem->rom_bank & 0x1F) | ((value & 0x03) << 5);
            else
                mem->ram_bank = value & 0x03;
        } else if (mem->mbc_type == MBC_3) {
            if (value <= 0x03) mem->ram_bank = value;
        } else if (mem->mbc_type == MBC_5) {
            mem->ram_bank = value & 0x0F;
        }
        return;
    }
    if (addr < 0x8000) { if (mem->mbc_type == MBC_1) mem->mbc1_mode = value & 0x01; return; }
    if (addr < 0xA000) { mem->vram[addr - 0x8000] = value; return; }
    if (addr < 0xC000) {
        if (mem->ram_enabled && mem->ext_ram) {
            uint8_t *p = ext_ram_ptr(mem, mem->ram_bank);
            if (p) p[addr - 0xA000] = value;
        }
        return;
    }
    if (addr < 0xE000) { mem->wram[addr - 0xC000] = value; return; }
    if (addr < 0xFE00) { mem->wram[(addr - 0xE000) & 0x1FFF] = value; return; }
    if (addr < 0xFEA0) { mem->oam[addr - 0xFE00] = value; return; }
    if (addr < 0xFF00) return;
    if (addr < 0xFF80) {
        uint8_t reg = (uint8_t)(addr - 0xFF00);
        switch (reg) {
            case IO_DIV:  gb->timer.div_internal = 0; mem->io[IO_DIV] = 0; return;
            case IO_TIMA: gb->timer.tima = value; return;
            case IO_TMA:  gb->timer.tma  = value; return;
            case IO_TAC:  gb->timer.tac  = value & 0x07; return;
            case IO_IF:   mem->io[IO_IF] = value | 0xE0; return;
            case IO_LY:   return;
            case IO_STAT: mem->io[IO_STAT] = (mem->io[IO_STAT] & 0x07) | (value & 0x78); return;
            case IO_DMA:
                mem->io[IO_DMA] = value;
                mem->dma_active = true;
                mem->dma_src    = (uint16_t)value << 8;
                mem->dma_index  = 0;
                return;
            default: mem->io[reg] = value; return;
        }
    }
    if (addr < 0xFFFF) { mem->hram[addr - 0xFF80] = value; return; }
    mem->IE = value;
}

void memory_dma_tick(GB *gb, int cycles) {
    Memory *mem = &gb->mem;
    if (!mem->dma_active) return;
    int ticks = cycles / 4;
    while (ticks-- > 0 && mem->dma_index < 160) {
        uint16_t src = mem->dma_src + (uint16_t)mem->dma_index;
        bool save = gb->tracer.enabled;
        gb->tracer.enabled = false;
        uint8_t val = memory_read(gb, src);
        gb->tracer.enabled = save;
        mem->oam[mem->dma_index++] = val;
    }
    if (mem->dma_index >= 160) mem->dma_active = false;
}
