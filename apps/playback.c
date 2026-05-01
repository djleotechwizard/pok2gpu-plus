#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <SDL2/SDL.h>

#include "../emu/gb.h"
#include "../emu/log.h"
#include "../gpu/rl.h"   /* RL_ACTION_MAP, RL_N_ACTIONS, RL_MAP_SENTINEL */

/* DMG grayscale palette: color index 0–3 → RGB */
static const uint8_t DMG_PALETTE[4][3] = {
    {255, 255, 255},
    {170, 170, 170},
    { 85,  85,  85},
    {  0,   0,   0},
};

/* Sequence file format versions */
#define SEQ_MAGIC_V3  0x42474C54u
#define SEQ_MAGIC_V2  0x42474C53u

/* Training-metric tile hash set — exact match to GPU kernel.
 * 4096 slots makes collisions negligible while keeping Knuth hash + linear probe. */
#define PB_TILE_HASH_BITS  12
#define PB_TILE_HASH_SIZE  (1 << PB_TILE_HASH_BITS)
#define PB_TILE_HASH_EMPTY 0xFFFFFFFFu

typedef struct {
    uint32_t slots[PB_TILE_HASH_SIZE];
    uint32_t n_unique;
    uint8_t  prev_map;
    bool     seeded;
} TileSet;

static void tileset_reset(TileSet *ts) {
    for (int i = 0; i < PB_TILE_HASH_SIZE; i++) ts->slots[i] = PB_TILE_HASH_EMPTY;
    ts->n_unique  = 0;
    ts->prev_map  = 0xFF;
    ts->seeded    = false;
}

/* Returns true if tile_id was newly inserted (not a duplicate). */
static bool tileset_insert(TileSet *ts, uint32_t tile_id) {
    uint32_t h = (tile_id * 2654435761u) >> (32 - PB_TILE_HASH_BITS);
    for (int p = 0; p < PB_TILE_HASH_SIZE; p++) {
        uint32_t k = ts->slots[h];
        if (k == tile_id)            return false;
        if (k == PB_TILE_HASH_EMPTY) { ts->slots[h] = tile_id; return true; }
        h = (h + 1) & (PB_TILE_HASH_SIZE - 1);
    }
    return false;
}

/* Score one action step — mirrors rl_episode_kernel exactly. */
static void tileset_sample(TileSet *ts, uint8_t cur_map, uint8_t py, uint8_t px,
                            bool verbose, uint32_t step,
                            uint8_t act_idx __attribute__((unused)),
                            const char *act_name) {
    if (!ts->seeded) {
        uint32_t start_id = ((uint32_t)ts->prev_map << 16) | RL_MAP_SENTINEL;
        tileset_insert(ts, start_id);
        ts->seeded = true;
    }

    if (cur_map != ts->prev_map) {
        uint32_t map_id = ((uint32_t)cur_map << 16) | RL_MAP_SENTINEL;
        if (tileset_insert(ts, map_id)) ts->n_unique++;
        ts->prev_map = cur_map;
    }

    uint32_t tile_id = ((uint32_t)cur_map << 16) | ((uint32_t)py << 8) | px;
    bool new_tile = tileset_insert(ts, tile_id);
    if (new_tile) ts->n_unique++;

    if (verbose) {
        fprintf(stderr, "step %4u  act=%-6s  map=%02X  y=%3u  x=%3u  score=%u%s\n",
            step, act_name, cur_map, py, px, ts->n_unique,
            new_tile ? "  *NEW*" : "");
    }
}

static const char *action_name(uint8_t idx) {
    static const char *names[] = {"none","up","down","left","right","A","B","start","select"};
    return (idx < RL_N_ACTIONS) ? names[idx] : "?";
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <rom.gb> <best_seq.bin>\n"
        "Options:\n"
        "  -s N   Window scale factor (default 3 = 480×432)\n"
        "  -l N   Loop N times; 0 = loop forever (default 0)\n"
        "  -x N   Speed multiplier (default 1 = real GB speed, 0 = unlimited)\n"
        "  -k     Skip intro frames (jump straight to RL portion)\n"
        "  -v     Verbose: print per-step state\n"
        "  -h     Show help\n",
        argv0);
}

int main(int argc, char **argv)
{
    const char *rom_path = NULL;
    const char *seq_path = NULL;
    int   scale       = 3;
    int   loops       = 0;
    int   speed       = 1;
    bool  skip_intro  = false;
    bool  verbose     = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-h"))             { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "-s") && i+1<argc) scale   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-l") && i+1<argc) loops   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-x") && i+1<argc) speed  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k"))             skip_intro = true;
        else if (!strcmp(argv[i], "-v"))             verbose    = true;
        else if (!rom_path) rom_path = argv[i];
        else if (!seq_path) seq_path = argv[i];
        else { LOG_ERR("unknown argument: %s", argv[i]); usage(argv[0]); return 1; }
    }
    if (!rom_path || !seq_path) { usage(argv[0]); return 1; }
    if (scale < 1) scale = 1;

    /* Load sequence file */
    FILE *sf = fopen(seq_path, "rb");
    if (!sf) { LOG_ERR("cannot open sequence: %s", seq_path); return 1; }

    uint32_t hdr[5] = {0};
    if (fread(hdr, sizeof(uint32_t), 4, sf) != 4) {
        LOG_ERR("sequence file too short"); fclose(sf); return 1;
    }
    uint32_t action_repeat = 1;
    if (hdr[0] == SEQ_MAGIC_V3) {
        if (fread(&hdr[4], sizeof(uint32_t), 1, sf) != 1) {
            LOG_ERR("truncated v3 header"); fclose(sf); return 1;
        }
        action_repeat = hdr[4];
    } else if (hdr[0] == SEQ_MAGIC_V2) {
        action_repeat = 1;
    } else {
        LOG_ERR("invalid or outdated sequence file (magic=0x%08X) — re-run rl_train", hdr[0]);
        fclose(sf); return 1;
    }

    uint32_t n_steps      = hdr[1];
    uint32_t best_score   = hdr[2];
    uint32_t intro_frames = hdr[3];
    uint32_t rl_steps     = n_steps - intro_frames;
    uint32_t rl_gb_frames = rl_steps * action_repeat;
    uint32_t total_gb_frames = intro_frames + rl_gb_frames;

    uint8_t *act_seq = (uint8_t *)malloc((size_t)n_steps);
    if (!act_seq) { LOG_ERR("OOM"); fclose(sf); return 1; }
    if (fread(act_seq, 1, n_steps, sf) != n_steps) {
        LOG_ERR("sequence file truncated"); free(act_seq); fclose(sf); return 1;
    }
    fclose(sf);

    LOG_INFO("loaded: %s  steps=%u  (intro=%u + rl=%u × %u = %u frames)",
        seq_path, n_steps, intro_frames, rl_steps, action_repeat, rl_gb_frames);
    LOG_INFO("gpu score=%u  scale=%dx  rom=%s", best_score, scale, rom_path);

    /* Initialise Game Boy */
    GB gb;
    gb_init(&gb);
    if (gb_load_rom(&gb, rom_path) != 0) {
        LOG_ERR("failed to load ROM: %s", rom_path);
        free(act_seq); return 1;
    }

    /* Initialise SDL2 */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        LOG_ERR("SDL_Init failed: %s", SDL_GetError());
        free(act_seq); return 1;
    }
    SDL_Window *win = SDL_CreateWindow(
        "pok2gpu — RL Best Agent",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        160 * scale, 144 * scale, 0);
    if (!win) {
        LOG_ERR("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit(); free(act_seq); return 1;
    }
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    SDL_RenderSetLogicalSize(ren, 160, 144);
    SDL_Texture *tex = SDL_CreateTexture(
        ren, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, 160, 144);
    if (!tex) {
        LOG_ERR("SDL_CreateTexture failed: %s", SDL_GetError());
        SDL_DestroyRenderer(ren); SDL_DestroyWindow(win);
        SDL_Quit(); free(act_seq); return 1;
    }

    bool running  = true;
    int  loop_num = 0;
    uint32_t train_score_final = 0;

    TileSet ts;
    char title_buf[128];

    while (running && (loops == 0 || loop_num < loops)) {
        gb_init(&gb);
        if (gb_load_rom(&gb, rom_path) != 0) break;

        tileset_reset(&ts);
        ts.prev_map = gb.mem.wram[0x135E];

        uint32_t next_tick = SDL_GetTicks();
        uint32_t frame_ms  = (speed > 0) ? (1000u / (60u * (uint32_t)speed)) : 0u;
        if (speed > 0 && frame_ms < 1u) frame_ms = 1u;

        for (uint32_t f = 0; f < total_gb_frames && running; f++) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                if (e.type == SDL_QUIT)                          { running = false; break; }
                if (e.type == SDL_KEYDOWN &&
                    e.key.keysym.sym == SDLK_ESCAPE)             { running = false; break; }
            }
            if (!running) break;

            uint32_t seq_idx;
            if (f < intro_frames) {
                seq_idx = f;
            } else {
                uint32_t rl_f = f - intro_frames;
                seq_idx = intro_frames + (rl_f / action_repeat);
            }
            uint8_t act_idx = (seq_idx < n_steps) ? act_seq[seq_idx] : 0u;
            gb.mem.joypad_action = (act_idx < RL_N_ACTIONS) ? RL_ACTION_MAP[act_idx] : 0u;

            gb_run_frames(&gb, 1);

            /* Sample tile at end of each RL action step — matches GPU kernel. */
            if (f >= intro_frames) {
                uint32_t rl_f = f - intro_frames;
                bool is_step_end = ((rl_f + 1) % action_repeat == 0);
                if (is_step_end) {
                    uint32_t step = rl_f / action_repeat;
                    uint8_t cur_map = gb.mem.wram[0x135E];
                    uint8_t py      = gb.mem.wram[0x1361];
                    uint8_t px      = gb.mem.wram[0x1362];
                    tileset_sample(&ts, cur_map, py, px, verbose, step,
                                   act_idx, action_name(act_idx));
                }
            }

            if (skip_intro && f < intro_frames) continue;

            void *pixels; int pitch;
            SDL_LockTexture(tex, NULL, &pixels, &pitch);
            uint8_t *dst = (uint8_t *)pixels;
            for (int i = 0; i < 160 * 144; i++) {
                const uint8_t *rgb = DMG_PALETTE[gb.ppu.framebuffer[i] & 3];
                dst[i*3+0] = rgb[0]; dst[i*3+1] = rgb[1]; dst[i*3+2] = rgb[2];
            }
            SDL_UnlockTexture(tex);
            SDL_RenderClear(ren);
            SDL_RenderCopy(ren, tex, NULL, NULL);
            SDL_RenderPresent(ren);

            /* Frame pacing: target ~60 fps × speed multiplier.
             * Resync if we fall behind by more than one frame (e.g. after -k skip). */
            if (speed > 0) {
                next_tick += frame_ms;
                uint32_t now = SDL_GetTicks();
                if (now > next_tick + frame_ms) next_tick = now;
                if (now < next_tick) SDL_Delay(next_tick - now);
            }

            if (f % 8 == 0) {
                bool in_intro = (f < intro_frames);
                snprintf(title_buf, sizeof(title_buf),
                    "pok2gpu RL — %s %u/%u  tiles=%u/%u  loop %d",
                    in_intro ? "INTRO" : "RL",
                    in_intro ? f : f - intro_frames,
                    in_intro ? intro_frames : rl_gb_frames,
                    ts.n_unique, best_score, loop_num + 1);
                SDL_SetWindowTitle(win, title_buf);
            }
        }

        train_score_final = ts.n_unique;
        fprintf(stderr, "loop %d  cpu=%u  gpu=%u  diff=%+d\n",
                loop_num + 1, train_score_final, best_score,
                (int)train_score_final - (int)best_score);

        loop_num++;
        if (loops != 0 && loop_num >= loops) break;
        SDL_Delay(500);
    }

    fprintf(stderr,
        "\n[info]  summary after %d loop(s)\n"
        "        gpu training score : %u\n"
        "        cpu replay score   : %u\n"
        "        difference         : %+d tiles\n",
        loop_num, best_score, train_score_final,
        (int)best_score - (int)train_score_final);

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    free(act_seq);
    return 0;
}
