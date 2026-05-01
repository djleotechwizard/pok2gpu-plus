#include <string.h>
#include <stdio.h>
#include "gb.h"

#define DOTS_PER_LINE  456
#define MODE2_END       80
#define MODE3_END      252
#define VBLANK_START   144
#define TOTAL_LINES    154

static const uint8_t DMG_PALETTE[4][3] = {
    {255, 255, 255},
    {170, 170, 170},
    { 85,  85,  85},
    {  0,   0,   0},
};

void ppu_init(PPU *ppu) {
    memset(ppu, 0, sizeof(*ppu));
    ppu->mode     = PPU_OAM;
    ppu->scanline = 0;
    ppu->dot      = 0;
    ppu->frame    = 0;
}

static inline void raise_int(GB *gb, uint8_t bit) {
    gb->mem.io[0x0F] |= bit;
}

#define INT_VBLANK 0x01
#define INT_STAT   0x02

static void check_stat_irq(GB *gb) {
    uint8_t stat = gb->mem.io[0x41];
    uint8_t lcdc = gb->mem.io[0x40];
    if (!(lcdc & 0x80)) return;

    bool fire = false;
    switch (gb->ppu.mode) {
        case PPU_HBLANK: fire = (stat & 0x08) != 0; break;
        case PPU_VBLANK: fire = (stat & 0x10) != 0; break;
        case PPU_OAM:    fire = (stat & 0x20) != 0; break;
        default: break;
    }
    if ((stat & 0x40) && (gb->ppu.scanline == (int)gb->mem.io[0x45]))
        fire = true;
    if (fire) raise_int(gb, INT_STAT);
}

static void render_scanline(GB *gb, int ly) {
    uint8_t lcdc = gb->mem.io[0x40];
    uint8_t *row = gb->ppu.framebuffer + ly * 160;
    uint8_t bg_raw[160];

    memset(row, 0, 160);
    memset(bg_raw, 0, sizeof(bg_raw));

    if (!(lcdc & 0x01)) return;

    uint8_t  scy = gb->mem.io[0x42];
    uint8_t  scx = gb->mem.io[0x43];
    uint8_t  bgp = gb->mem.io[0x47];

    uint16_t map_base = (lcdc & 0x08) ? 0x9C00 : 0x9800;
    bool     unsigned_addr = (lcdc & 0x10) != 0;

    int wy = (ly + scy) & 0xFF;
    int tile_row = wy >> 3;
    int tile_py  = wy & 7;

    for (int x = 0; x < 160; x++) {
        int wx       = (x + scx) & 0xFF;
        int tile_col = wx >> 3;
        int tile_px  = wx & 7;

        uint16_t map_addr = map_base + (uint16_t)(tile_row * 32 + tile_col);
        uint8_t  tile_idx = gb->mem.vram[map_addr - 0x8000];

        uint16_t tile_addr;
        if (unsigned_addr) {
            tile_addr = 0x8000 + (uint16_t)tile_idx * 16;
        } else {
            tile_addr = (uint16_t)(0x9000 + (int8_t)tile_idx * 16);
        }

        uint16_t row_off = tile_addr - 0x8000 + (uint16_t)(tile_py * 2);
        uint8_t  lo = gb->mem.vram[row_off];
        uint8_t  hi = gb->mem.vram[row_off + 1];

        int bit   = 7 - tile_px;
        int cidx  = (((hi >> bit) & 1) << 1) | ((lo >> bit) & 1);

        row[x]    = (bgp >> (cidx * 2)) & 0x03;
        bg_raw[x] = (uint8_t)cidx;
    }

    if ((lcdc & 0x20) && ly >= (int)gb->mem.io[0x4A]) {
        int wx  = (int)gb->mem.io[0x4B] - 7;
        if (wx < 160) {
            uint16_t win_map = (lcdc & 0x40) ? 0x9C00u : 0x9800u;
            int win_y      = ly - (int)gb->mem.io[0x4A];
            int win_trow   = win_y >> 3;
            int win_tpy    = win_y & 7;

            for (int x = (wx < 0 ? 0 : wx); x < 160; x++) {
                int win_x    = x - wx;
                int win_tcol = win_x >> 3;
                int win_tpx  = win_x & 7;

                uint16_t maddr  = win_map + (uint16_t)(win_trow * 32 + win_tcol);
                uint8_t  tidx   = gb->mem.vram[maddr - 0x8000];

                uint16_t taddr;
                if (unsigned_addr) {
                    taddr = 0x8000u + (uint16_t)tidx * 16;
                } else {
                    taddr = (uint16_t)(0x9000 + (int8_t)tidx * 16);
                }

                uint16_t roff = taddr - 0x8000u + (uint16_t)(win_tpy * 2);
                uint8_t  lo2  = gb->mem.vram[roff];
                uint8_t  hi2  = gb->mem.vram[roff + 1];

                int bit2 = 7 - win_tpx;
                int cidx = (((hi2 >> bit2) & 1) << 1) | ((lo2 >> bit2) & 1);
                row[x]    = (bgp >> (cidx * 2)) & 0x03;
                bg_raw[x] = (uint8_t)cidx;
            }
        }
    }

    if (lcdc & 0x02) {
        int sprite_h = (lcdc & 0x04) ? 16 : 8;
        int n = 0;
        for (int s = 0; s < 40 && n < 10; s++) {
            int sy = (int)gb->mem.oam[s * 4 + 0] - 16;
            if (ly < sy || ly >= sy + sprite_h) continue;
            n++;

            int     sx     = (int)gb->mem.oam[s * 4 + 1] - 8;
            uint8_t ti     = gb->mem.oam[s * 4 + 2];
            uint8_t fl     = gb->mem.oam[s * 4 + 3];
            uint8_t obp    = (fl & 0x10) ? gb->mem.io[0x49] : gb->mem.io[0x48];
            bool    bg_pri = (fl & 0x80) != 0;

            if (sprite_h == 16) ti &= 0xFE;
            int py = ly - sy;
            if (fl & 0x40) py = sprite_h - 1 - py;

            uint16_t roff = (uint16_t)ti * 16 + (uint16_t)(py * 2);
            uint8_t  lo   = gb->mem.vram[roff];
            uint8_t  hi   = gb->mem.vram[roff + 1];

            for (int px = 0; px < 8; px++) {
                int screen_x = sx + px;
                if (screen_x < 0 || screen_x >= 160) continue;
                int bit_pos = (fl & 0x20) ? px : (7 - px);
                int cidx = (((hi >> bit_pos) & 1) << 1) | ((lo >> bit_pos) & 1);
                if (cidx == 0) continue;
                if (bg_pri && bg_raw[screen_x] != 0) continue;
                row[screen_x] = (obp >> (cidx * 2)) & 0x03;
            }
        }
    }
}

void ppu_save_ppm(const PPU *ppu, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write PPM: %s\n", path); return; }
    fprintf(f, "P6\n160 144\n255\n");
    for (int i = 0; i < 160 * 144; i++) {
        const uint8_t *rgb = DMG_PALETTE[ppu->framebuffer[i] & 3];
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

void ppu_step(GB *gb, int cycles) {
    uint8_t lcdc = gb->mem.io[0x40];
    if (!(lcdc & 0x80)) {
        gb->ppu.dot += cycles;
        if (gb->ppu.dot >= DOTS_PER_LINE * TOTAL_LINES) {
            gb->ppu.dot -= DOTS_PER_LINE * TOTAL_LINES;
            gb->ppu.frame++;
        }
        gb->ppu.scanline = 0;
        gb->mem.io[0x44] = 0;
        gb->mem.io[0x41] = (gb->mem.io[0x41] & 0xF8) | PPU_HBLANK;
        gb->ppu.mode     = PPU_HBLANK;
        return;
    }

    gb->ppu.dot += cycles;

    while (gb->ppu.dot >= DOTS_PER_LINE) {
        gb->ppu.dot -= DOTS_PER_LINE;
        gb->ppu.scanline++;

        if (gb->ppu.scanline == VBLANK_START) {
            gb->ppu.mode     = PPU_VBLANK;
            gb->mem.io[0x41] = (gb->mem.io[0x41] & 0xF8) | PPU_VBLANK;
            raise_int(gb, INT_VBLANK);
            check_stat_irq(gb);
            gb->ppu.frame++;
        } else if (gb->ppu.scanline >= TOTAL_LINES) {
            gb->ppu.scanline = 0;
            gb->ppu.mode     = PPU_OAM;
            gb->mem.io[0x41] = (gb->mem.io[0x41] & 0xF8) | PPU_OAM;
            check_stat_irq(gb);
        }

        gb->mem.io[0x44] = (uint8_t)gb->ppu.scanline;

        bool coinc = (gb->ppu.scanline == (int)gb->mem.io[0x45]);
        if (coinc) gb->mem.io[0x41] |=  0x04;
        else       gb->mem.io[0x41] &= ~0x04;
        if (coinc && (gb->mem.io[0x41] & 0x40)) raise_int(gb, INT_STAT);
    }

    if (gb->ppu.scanline < VBLANK_START) {
        PPUMode new_mode;
        if      (gb->ppu.dot < MODE2_END)  new_mode = PPU_OAM;
        else if (gb->ppu.dot < MODE3_END)  new_mode = PPU_DRAW;
        else                               new_mode = PPU_HBLANK;

        if (new_mode != gb->ppu.mode) {
            if (new_mode == PPU_HBLANK)
                render_scanline(gb, gb->ppu.scanline);
            gb->ppu.mode     = new_mode;
            gb->mem.io[0x41] = (gb->mem.io[0x41] & 0xF8) | (uint8_t)new_mode;
            check_stat_irq(gb);
        }
    }
}
