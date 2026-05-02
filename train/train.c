#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "../emu/gb.h"
#include "batch.h"
#include "../emu/log.h"
#include "../gpu/batch.h"
#include "../gpu/rl.h"
#include "../analysis/ir_interp.h"

/* Training defaults */
#define DEFAULT_N_STATES      2048
#define DEFAULT_N_GENERATIONS   10
#define DEFAULT_EP_FRAMES      600
#define DEFAULT_ACTION_REPEAT    8   /* GB frames per action decision (one tile ≈ 8 frames) */
#define DEFAULT_INTRO_FRAMES  3600   /* frames of scripted intro to simulate */
#define DEFAULT_N_ELITE         64
#define DEFAULT_MUTATION_RATE  0.05f
#define DEFAULT_OUTPUT        "best_seq.bin"

/* Weighted action table for Pokémon Red overworld.
 * Heavily favours movement; excludes start (opens pause menu).
 * up/down/left/right 20% each, A 10%, nothing 5%, B 5% */
static const uint8_t WEIGHTED_ACTIONS[20] = {
    1,1,1,1,  /* up    ×4 → 20% */
    2,2,2,2,  /* down  ×4 → 20% */
    3,3,3,3,  /* left  ×4 → 20% */
    4,4,4,4,  /* right ×4 → 20% */
    5,5,       /* A     ×2 → 10% */
    0,         /* nothing×1→  5% */
    6,         /* B     ×1 →  5% */
};
#define N_WEIGHTED_ACTIONS 20

static double clock_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_desc(const void *a, const void *b, void *arg) {
    const uint32_t *counts = (const uint32_t *)arg;
    uint32_t ca = counts[*(const int *)a];
    uint32_t cb = counts[*(const int *)b];
    return (cb > ca) - (cb < ca);
}

static const char *fmt_dur(double secs, char *buf, int bufsz) {
    if      (secs < 60)   snprintf(buf, bufsz, "%.0fs",  secs);
    else if (secs < 3600) snprintf(buf, bufsz, "%dm%02.0fs",
                                   (int)(secs/60), fmod(secs, 60));
    else                  snprintf(buf, bufsz, "%dh%02dm",
                                   (int)(secs/3600), (int)(fmod(secs/60, 60)));
    return buf;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] <rom.gb>\n"
        "Options:\n"
        "  -n N    Parallel agents          (default %d)\n"
        "  -g G    Generations              (default %d)\n"
        "  -f F    GB frames per episode    (default %d)\n"
        "  -r R    Action repeat (frames)   (default %d; 1=per-frame)\n"
        "  -w W    Scripted intro frames    (default %d; 0=skip)\n"
        "  -e E    Elite survivors          (default %d)\n"
        "  -m M    Mutation rate 0..1       (default %.2f)\n"
        "  -o FILE Output sequence file     (default %s)\n"
        "  -s SEED Random seed              (default time)\n"
        "  -P N    Profile N generations then write <output>.profile\n"
        "          (skips RL training; feed output to trace_gen to build JIT)\n"
        "  -H F    Successive Halving: cull bottom 50%% after F frames then\n"
        "          continue top 50%% for remaining frames. F < ep_frames.\n"
        "  -V N    CPU verification: replay best sequence on CPU every N gens\n"
        "          and print GPU vs CPU score diff.\n"
        "  -h      Show help\n",
        argv0,
        DEFAULT_N_STATES, DEFAULT_N_GENERATIONS, DEFAULT_EP_FRAMES,
        DEFAULT_ACTION_REPEAT, DEFAULT_INTRO_FRAMES,
        DEFAULT_N_ELITE, DEFAULT_MUTATION_RATE, DEFAULT_OUTPUT);
}

/* Sequence file format (v3):
 *   uint32_t magic         = 0x42474C54
 *   uint32_t n_steps       = intro_frames + action_steps
 *   uint32_t best_score
 *   uint32_t intro_frames
 *   uint32_t action_repeat  (GB frames per RL action step; 1 = legacy)
 *   uint8_t  seq[n_steps]   (intro: 1 byte/frame; RL: 1 byte/action_step)
 *
 * Playback: intro frames run 1:1, RL steps each held for action_repeat frames. */
#define SEQ_MAGIC 0x42474C54u

/* Build the intro action-index sequence mirroring run_pokered_intro(). */
static void build_intro_seq(uint8_t *out, int n_frames)
{
    for (int f = 0; f < n_frames; f++) {
        uint8_t idx = 0;
        if      (f == 400)               idx = 5;  /* A: open title menu  */
        else if (f == 413)               idx = 5;  /* A: select NEW GAME  */
        else if (f >= 414 && f % 2 == 0) idx = 5;  /* A: spam all text    */
        out[f] = idx;
    }
}

static void save_seq(const char *path,
                     const uint8_t *intro_seq, int intro_frames,
                     const uint8_t *rl_seq,    int action_steps,
                     int action_repeat,         uint32_t score)
{
    FILE *f = fopen(path, "wb");
    if (!f) { LOG_ERR("cannot write %s", path); return; }
    uint32_t n_steps = (uint32_t)(intro_frames + action_steps);
    uint32_t hdr[5] = { SEQ_MAGIC, n_steps, score,
                        (uint32_t)intro_frames, (uint32_t)action_repeat };
    fwrite(hdr, sizeof(uint32_t), 5, f);
    if (intro_frames > 0) fwrite(intro_seq, 1, (size_t)intro_frames, f);
    fwrite(rl_seq, 1, (size_t)action_steps, f);
    fclose(f);
    LOG_SAVE("score=%u  steps=%d  repeat=%d  → %s",
             score, action_steps, action_repeat, path);
}

/*
 * Run the scripted Pokémon Red intro on a single CPU GB instance.
 * After this the player is in Pallet Town bedroom, ready for RL.
 *
 * Input timing:
 *   frames 0–399    no input  (logo + title animation)
 *   frame 400       A         opens title menu (cursor on NEW GAME)
 *   frame 413       A         selects NEW GAME
 *   frame 414+      A/2fr     advances Oak's speech, names, final transition
 *
 * Total intro: ~800–1200 frames. The default ceiling (3600) is generous.
 * After return, states[0] is ready to clone to all GPU agents.
 */
static void run_pokered_intro(GB *gb, int max_frames)
{
    /* Pokémon Red WRAM addresses (index = GB_addr - 0xC000):
     *   wCurMap  0xD35E → 0x135E
     *   wYCoord  0xD361 → 0x1361
     *   wXCoord  0xD362 → 0x1362  */
    LOG_INFO("intro: running %d frames  (A→menu → A→NEW GAME → A-spam text)", max_frames);

    for (int f = 0; f < max_frames; f++) {
        uint8_t action = 0x00;
        if      (f == 400)               action = 0x01;
        else if (f == 413)               action = 0x01;
        else if (f >= 414 && f % 2 == 0) action = 0x01;

        gb->mem.joypad_action = action;
        gb_run_frames(gb, 1);

        if ((f + 1) % 150 == 0 || f == max_frames - 1)
            LOG_PROG("intro  %4d/%-4d  PC=%04X  map=%02X  pos=(%3d,%3d)       ",
                f + 1, max_frames,
                gb->cpu.PC,
                gb->mem.wram[0x135E],
                gb->mem.wram[0x1362],
                gb->mem.wram[0x1361]);
    }
    fprintf(stderr, "\n");
}

/* CPU replay: mirrors rl_episode_kernel exactly.
 * Same Knuth hash set (4096 slots), same map sentinel,
 * once-per-action-step sampling at step end. */
#define CPU_TILE_HASH_BITS  12
#define CPU_TILE_HASH_SIZE  (1 << CPU_TILE_HASH_BITS)
#define CPU_TILE_HASH_EMPTY 0xFFFFFFFFu

static bool cpu_tile_insert(uint32_t *slots, uint32_t tile_id) {
    uint32_t h = (tile_id * 2654435761u) >> (32 - CPU_TILE_HASH_BITS);
    for (int p = 0; p < CPU_TILE_HASH_SIZE; p++) {
        uint32_t k = slots[h];
        if (k == tile_id)             return false;
        if (k == CPU_TILE_HASH_EMPTY) { slots[h] = tile_id; return true; }
        h = (h + 1) & (CPU_TILE_HASH_SIZE - 1);
    }
    return false;
}

static uint32_t cpu_score_sequence(
    const GB      *tmpl,
    GBBatch       *batch,
    const uint8_t *ext_ram_tmpl, uint32_t ext_ram_size,
    const uint8_t *rl_seq, int action_steps, int action_repeat)
{
    GB gb = *tmpl;
    gb.mem.rom = batch->shared_rom;

    uint8_t *ext_ram_buf = NULL;
    if (ext_ram_size > 0 && ext_ram_tmpl) {
        ext_ram_buf = (uint8_t *)malloc(ext_ram_size);
        if (ext_ram_buf) {
            memcpy(ext_ram_buf, ext_ram_tmpl, ext_ram_size);
            gb.mem.ext_ram = ext_ram_buf;
        }
    } else {
        gb.mem.ext_ram = NULL;
    }

    uint32_t slots[CPU_TILE_HASH_SIZE];
    for (int i = 0; i < CPU_TILE_HASH_SIZE; i++) slots[i] = CPU_TILE_HASH_EMPTY;
    uint32_t n_unique = 0;
    uint8_t  prev_map = gb.mem.wram[0x135E];

    cpu_tile_insert(slots, ((uint32_t)prev_map << 16) | 0xFFFFu);

    for (int step = 0; step < action_steps; step++) {
        uint8_t act = rl_seq[step];
        gb.mem.joypad_action = (act < RL_N_ACTIONS) ? RL_ACTION_MAP[act] : 0u;
        uint32_t target_frame = gb.ppu.frame + (uint32_t)action_repeat;
        while (gb.ppu.frame < target_frame) {
            int cyc = ir_step(&gb, &batch->cache, batch->cfg);
            ppu_step(&gb, cyc);
            timer_step(&gb, cyc);
            memory_dma_tick(&gb, cyc);
        }

        uint8_t cur_map = gb.mem.wram[0x135E];
        uint8_t py      = gb.mem.wram[0x1361];
        uint8_t px      = gb.mem.wram[0x1362];

        if (cur_map != prev_map) {
            if (cpu_tile_insert(slots, ((uint32_t)cur_map << 16) | 0xFFFFu)) n_unique++;
            prev_map = cur_map;
        }
        if (cpu_tile_insert(slots, ((uint32_t)cur_map << 16) | ((uint32_t)py << 8) | px))
            n_unique++;
    }

    free(ext_ram_buf);
    return n_unique;
}

int main(int argc, char **argv)
{
    const char  *rom_path     = NULL;
    const char  *out_path     = DEFAULT_OUTPUT;
    int          n_states     = DEFAULT_N_STATES;
    int          n_generations= DEFAULT_N_GENERATIONS;
    int          ep_frames    = DEFAULT_EP_FRAMES;
    int          action_repeat= DEFAULT_ACTION_REPEAT;
    int          intro_frames = DEFAULT_INTRO_FRAMES;
    int          n_elite      = DEFAULT_N_ELITE;
    float        mutation_rate= DEFAULT_MUTATION_RATE;
    unsigned int seed         = (unsigned int)time(NULL);
    int          n_profile    = 0;
    int          halving_frames  = 0;
    int          verify_interval = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-h"))             { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "-n") && i+1<argc)   n_states      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i+1<argc)   n_generations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-f") && i+1<argc)   ep_frames     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i+1<argc)   action_repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i+1<argc)   intro_frames  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-e") && i+1<argc)   n_elite       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i+1<argc)   mutation_rate = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1<argc)   out_path      = argv[++i];
        else if (!strcmp(argv[i], "-s") && i+1<argc)   seed          = (unsigned int)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-P") && i+1<argc)   n_profile       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H") && i+1<argc)   halving_frames  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-V") && i+1<argc)   verify_interval = atoi(argv[++i]);
        else if (argv[i][0] != '-') rom_path = argv[i];
        else { LOG_ERR("unknown option: %s", argv[i]); usage(argv[0]); return 1; }
    }
    if (!rom_path) { usage(argv[0]); return 1; }
    if (action_repeat < 1) action_repeat = 1;
    int action_steps = ep_frames / action_repeat;
    if (action_steps < 1) action_steps = 1;
    if (n_elite >= n_states) n_elite = n_states / 4;
    if (halving_frames > 0 && halving_frames >= ep_frames) {
        LOG_WARN("-H %d >= ep_frames %d — disabling halving", halving_frames, ep_frames);
        halving_frames = 0;
    }

    srand(seed);
    LOG_INFO("rom=%s  agents=%d  seed=%u", rom_path, n_states, seed);

    /* Build CPU batch: loads ROM, builds CFG/IR, allocates N state slots */
    double t0 = clock_s();
    GBBatch *cpu_batch = batch_create(n_states, rom_path);
    if (!cpu_batch) { LOG_ERR("failed to create CPU batch"); return 1; }
    LOG_INFO("cpu batch ready  (%.2fs)", clock_s() - t0);

    uint32_t ext_ram_size = cpu_batch->ext_ram_size;

    /* Run scripted intro on states[0], then clone to all slots */
    if (intro_frames > 0) {
        t0 = clock_s();
        run_pokered_intro(&cpu_batch->states[0], intro_frames);
        GB *s0 = &cpu_batch->states[0];
        LOG_INFO("intro done  %.2fs  PC=%04X  frame=%u  map=%02X  pos=(%d,%d)",
            clock_s() - t0, s0->cpu.PC, s0->ppu.frame,
            s0->mem.wram[0x135E], s0->mem.wram[0x1362], s0->mem.wram[0x1361]);

        for (int i = 1; i < n_states; i++) {
            uint8_t *per_inst_ext_ram = cpu_batch->states[i].mem.ext_ram;
            cpu_batch->states[i] = *s0;
            cpu_batch->states[i].mem.rom     = cpu_batch->shared_rom;
            cpu_batch->states[i].mem.ext_ram = per_inst_ext_ram;
            if (ext_ram_size > 0 && per_inst_ext_ram)
                memcpy(per_inst_ext_ram, s0->mem.ext_ram, ext_ram_size);
        }
        LOG_INFO("cloned post-intro state to all %d agents", n_states);
    }

    /* Pack per-instance ext_ram into a flat host array for GPU upload */
    uint8_t *host_ext_rams = NULL;
    if (ext_ram_size > 0) {
        host_ext_rams = (uint8_t *)malloc((size_t)n_states * ext_ram_size);
        if (!host_ext_rams) { LOG_ERR("OOM"); return 1; }
        for (int i = 0; i < n_states; i++)
            memcpy(host_ext_rams + (size_t)i * ext_ram_size,
                   cpu_batch->states[i].mem.ext_ram, ext_ram_size);
    }

    /* Build GPU batch from post-intro states */
    t0 = clock_s();
    GPUBatch *gpu_batch = gpu_batch_create(
        cpu_batch->cfg, cpu_batch->states,
        host_ext_rams, n_states,
        cpu_batch->rom_size, ext_ram_size);
    if (!gpu_batch) { LOG_ERR("failed to create GPU batch"); return 1; }
    LOG_INFO("gpu batch ready  (%.2fs)", clock_s() - t0);
    gpu_batch_print_stats(gpu_batch);

    /* Profiling mode: run N gens counting block hits, write .profile, exit */
    if (n_profile > 0) {
        size_t plen = strlen(out_path) + 9;
        char *profile_path = (char *)malloc(plen);
        if (!profile_path) { LOG_ERR("OOM"); return 1; }
        snprintf(profile_path, plen, "%s.profile", out_path);

        LOG_INFO("profiling mode: %d gen(s)  →  %s", n_profile, profile_path);

        RLBatch *rl_prof = rl_batch_create(gpu_batch, ep_frames, action_repeat);
        if (!rl_prof) { LOG_ERR("failed to create RL batch for profiling"); return 1; }

        int prof_action_steps = ep_frames / action_repeat;
        if (prof_action_steps < 1) prof_action_steps = 1;

        size_t seq_total_p = (size_t)n_states * prof_action_steps;
        uint8_t *seqs_p = (uint8_t *)malloc(seq_total_p);
        if (!seqs_p) { LOG_ERR("OOM"); return 1; }
        for (size_t i = 0; i < seq_total_p; i++)
            seqs_p[i] = WEIGHTED_ACTIONS[rand() % N_WEIGHTED_ACTIONS];
        rl_batch_set_seqs(rl_prof, seqs_p);

        gpu_batch_start_profiling(gpu_batch);

        double t_prof = clock_s();
        for (int pg = 0; pg < n_profile; pg++) {
            LOG_PROG("profiling  %d/%d ...", pg + 1, n_profile);
            fflush(stderr);
            rl_episode_run(rl_prof);
        }
        fprintf(stderr, "\n");

        gpu_batch_stop_profiling(gpu_batch);
        gpu_batch_dump_profile(gpu_batch, profile_path);

        LOG_INFO("profiling done  %.2fs", clock_s() - t_prof);
        fprintf(stderr,
            "\nNext steps to enable JIT hot-trace acceleration:\n"
            "  1. make trace_gen\n"
            "  2. ./trace_gen %s %s\n"
            "  3. make rl\n\n",
            profile_path, rom_path);

        free(seqs_p);
        free(profile_path);
        rl_batch_free(rl_prof);
        gpu_batch_free(gpu_batch);
        free(host_ext_rams);
        batch_free(cpu_batch);
        return 0;
    }

    /* Create RL batch — snapshots current GPU state as the episode reset point */
    RLBatch *rl = rl_batch_create(gpu_batch, ep_frames, action_repeat);
    if (!rl) { LOG_ERR("failed to create RL batch"); return 1; }

    /* Successive Halving: short batch for early culling (if -H was given) */
    RLBatch  *rl_short    = NULL;
    uint32_t *sh_counts   = NULL;
    uint8_t  *alive_mask  = NULL;
    uint8_t  *short_seqs  = NULL;
    int       halving_steps = 0;
    if (halving_frames > 0) {
        rl_short = rl_batch_create(gpu_batch, halving_frames, action_repeat);
        if (!rl_short) { LOG_ERR("failed to create short RL batch"); return 1; }
        halving_steps = rl_short->action_steps;
        sh_counts   = (uint32_t *)malloc((size_t)n_states * sizeof(uint32_t));
        alive_mask  = (uint8_t  *)malloc((size_t)n_states);
        short_seqs  = (uint8_t  *)malloc((size_t)n_states * halving_steps);
        if (!sh_counts || !alive_mask || !short_seqs) { LOG_ERR("OOM"); return 1; }
        LOG_INFO("successive halving: short=%d frames (%d steps) → cull 50%% → full eval",
                 halving_frames, halving_steps);
    }

    /* Allocate host buffers for evolutionary loop */
    size_t seq_total = (size_t)n_states * action_steps;
    uint8_t  *seqs     = (uint8_t  *)malloc(seq_total);
    uint8_t  *new_seqs = (uint8_t  *)malloc(seq_total);
    uint32_t *counts   = (uint32_t *)malloc((size_t)n_states * sizeof(uint32_t));
    int      *ranking  = (int      *)malloc((size_t)n_states * sizeof(int));
    uint8_t  *best_seq = (uint8_t  *)malloc((size_t)action_steps);
    if (!seqs || !new_seqs || !counts || !ranking || !best_seq) {
        LOG_ERR("OOM"); return 1;
    }

    uint8_t *intro_seq = (uint8_t *)calloc(intro_frames > 0 ? (size_t)intro_frames : 1, 1);
    if (!intro_seq) { LOG_ERR("OOM"); return 1; }
    if (intro_frames > 0) build_intro_seq(intro_seq, intro_frames);

    for (size_t i = 0; i < seq_total; i++)
        seqs[i] = WEIGHTED_ACTIONS[rand() % N_WEIGHTED_ACTIONS];
    rl_batch_set_seqs(rl, seqs);

    uint32_t best_score  = 0;
    int      stagnant    = 0;
    float    cur_mut     = mutation_rate;
    memset(best_seq, 0, action_steps);

    const GB      *post_intro_tmpl   = &cpu_batch->states[0];
    const uint8_t *post_intro_extram = cpu_batch->states[0].mem.ext_ram;
    uint32_t cpu_score_last = 0;

    fprintf(stderr,
        "\n[info]  training: agents=%d  ep=%d frames  steps=%d  repeat=%d  gens=%d\n"
        "[info]  elite=%d  mutation=%.2f  crossover=50%%  output=%s%s\n\n",
        n_states, ep_frames, action_steps, action_repeat, n_generations,
        n_elite, mutation_rate, out_path,
        halving_frames > 0 ? "  halving=on" : "");

#define ETA_WIN 10
    double ep_window[ETA_WIN];
    int    ep_win_n = 0;

    double t_train = clock_s();

    for (int gen = 0; gen < n_generations; gen++) {

        {
            char ebuf[32];
            double elapsed = clock_s() - t_train;
            LOG_PROG("gen %4d/%d  [running...]  elapsed=%s                    ",
                gen + 1, n_generations, fmt_dur(elapsed, ebuf, sizeof(ebuf)));
            fflush(stderr);
        }

        double tg = clock_s();

        if (rl_short) {
            /* Short eval: identify weak sequences early */
            for (int i = 0; i < n_states; i++)
                memcpy(short_seqs + (size_t)i * halving_steps,
                       seqs       + (size_t)i * action_steps,
                       halving_steps);
            rl_batch_set_seqs(rl_short, short_seqs);
            rl_episode_run(rl_short);
            rl_episode_get_counts(rl_short);
            memcpy(sh_counts, rl_short->h_tile_counts, (size_t)n_states * sizeof(uint32_t));

            /* Build alive mask: top 50% survive */
            int n_alive = n_states / 2;
            int *sh_rank = ranking;
            for (int i = 0; i < n_states; i++) sh_rank[i] = i;
            qsort_r(sh_rank, (size_t)n_states, sizeof(int), cmp_desc, sh_counts);
            memset(alive_mask, 0, (size_t)(unsigned)n_states);
            for (int i = 0; i < n_alive; i++) alive_mask[sh_rank[i]] = 1;
            rl_episode_set_alive(rl, alive_mask);

            /* Full eval: only top 50% run */
            rl_episode_run(rl);
            rl_episode_set_alive(rl, NULL);

            rl_episode_get_counts(rl);
            for (int i = 0; i < n_states; i++) {
                if (!alive_mask[i])
                    rl->h_tile_counts[i] = sh_counts[i];
            }
        } else {
            rl_episode_run(rl);
            rl_episode_get_counts(rl);
        }

        memcpy(counts, rl->h_tile_counts, (size_t)n_states * sizeof(uint32_t));
        double ep_time = clock_s() - tg;

        ep_window[ep_win_n % ETA_WIN] = ep_time;
        ep_win_n++;
        int win = (ep_win_n < ETA_WIN) ? ep_win_n : ETA_WIN;
        double avg = 0;
        for (int k = 0; k < win; k++) avg += ep_window[k];
        avg /= win;
        double eta = avg * (n_generations - gen - 1);

        for (int i = 0; i < n_states; i++) ranking[i] = i;
        qsort_r(ranking, (size_t)n_states, sizeof(int), cmp_desc, counts);

        uint32_t mx  = counts[ranking[0]];
        uint32_t mn  = counts[ranking[n_states - 1]];
        uint32_t p75 = counts[ranking[n_states / 4]];
        uint32_t p25 = counts[ranking[n_states * 3 / 4]];
        double   mean = 0;
        for (int i = 0; i < n_states; i++) mean += counts[i];
        mean /= n_states;

        double thr = (double)n_states * ep_frames / ep_time;

        bool improved = (mx > best_score);
        if (improved) {
            best_score = mx;
            stagnant   = 0;
            cur_mut    = mutation_rate;
            memcpy(best_seq, seqs + (size_t)ranking[0] * action_steps, action_steps);
            save_seq(out_path, intro_seq, intro_frames,
                     best_seq, action_steps, action_repeat, best_score);
        } else {
            stagnant++;
            /* After 3 stagnant gens ramp mutation to escape local optima.
             * Multiplies by 1.5× per stagnant gen, capped at 0.50. */
            if (stagnant >= 3)
                cur_mut = fminf(mutation_rate * powf(1.5f, (float)(stagnant - 2)), 0.50f);
        }

        if (verify_interval > 0 && (gen + 1) % verify_interval == 0) {
            cpu_score_last = cpu_score_sequence(
                post_intro_tmpl,
                cpu_batch,
                post_intro_extram, ext_ram_size,
                best_seq, action_steps, action_repeat);
        }

        {
            char ebuf[32], etabuf[32];
            double elapsed = clock_s() - t_train;
            if (verify_interval > 0 && (gen + 1) % verify_interval == 0) {
                fprintf(stderr,
                    "\rgen %4d/%d  "
                    "max=%4u  mean=%5.1f  p75=%4u  p25=%4u  min=%4u  "
                    "ep=%5.1fs  %5.0fk sf/s  mut=%.2f  "
                    "elapsed=%-8s  eta=%-8s  "
                    "gpu=%u  cpu=%u  diff=%+d"
                    "%s\n",
                    gen + 1, n_generations,
                    mx, mean, p75, p25, mn,
                    ep_time, thr / 1000.0, cur_mut,
                    fmt_dur(elapsed, ebuf, sizeof(ebuf)),
                    fmt_dur(eta,     etabuf, sizeof(etabuf)),
                    best_score, cpu_score_last,
                    (int)best_score - (int)cpu_score_last,
                    improved ? "  *** NEW BEST ***" : "");
            } else {
                fprintf(stderr,
                    "\rgen %4d/%d  "
                    "max=%4u  mean=%5.1f  p75=%4u  p25=%4u  min=%4u  "
                    "ep=%5.1fs  %5.0fk sf/s  mut=%.2f  "
                    "elapsed=%-8s  eta=%-8s"
                    "%s\n",
                    gen + 1, n_generations,
                    mx, mean, p75, p25, mn,
                    ep_time, thr / 1000.0, cur_mut,
                    fmt_dur(elapsed, ebuf, sizeof(ebuf)),
                    fmt_dur(eta,     etabuf, sizeof(etabuf)),
                    improved ? "  *** NEW BEST ***" : "");
            }
            fflush(stderr);
        }

        /* Next generation: elite → crossover → mutation */
        memcpy(new_seqs, seqs + (size_t)ranking[0] * action_steps, action_steps);

        for (int i = 1; i < n_elite; i++)
            memcpy(new_seqs + (size_t)i * action_steps,
                   seqs     + (size_t)ranking[i] * action_steps,
                   action_steps);

        int crossover_end = n_elite + (n_states - n_elite) / 2;
        for (int i = n_elite; i < n_states; i++) {
            uint8_t *dst = new_seqs + (size_t)i * action_steps;

            if (i < crossover_end) {
                int p1 = ranking[rand() % n_elite];
                int p2 = ranking[rand() % n_elite];
                int split = (int)((unsigned)rand() % (unsigned)action_steps);
                memcpy(dst,         seqs + (size_t)p1 * action_steps, split);
                memcpy(dst + split, seqs + (size_t)p2 * action_steps + split,
                       action_steps - split);
            } else {
                int parent = ranking[rand() % n_elite];
                memcpy(dst, seqs + (size_t)parent * action_steps, action_steps);
            }

            for (int t = 0; t < action_steps; t++)
                if ((float)rand() / (float)RAND_MAX < cur_mut)
                    dst[t] = WEIGHTED_ACTIONS[rand() % N_WEIGHTED_ACTIONS];
        }

        uint8_t *tmp = seqs; seqs = new_seqs; new_seqs = tmp;
        rl_batch_set_seqs(rl, seqs);
    }

    {
        char ebuf[32];
        LOG_INFO("done  %s  best=%u tiles  saved to %s",
                 fmt_dur(clock_s() - t_train, ebuf, sizeof(ebuf)),
                 best_score, out_path);
    }

    free(intro_seq);
    free(seqs); free(new_seqs); free(counts); free(ranking); free(best_seq);
    free(sh_counts); free(alive_mask); free(short_seqs);
    (void)ep_frames;
    free(host_ext_rams);
    rl_batch_free(rl_short);
    rl_batch_free(rl);
    gpu_batch_free(gpu_batch);
    batch_free(cpu_batch);
    return 0;
}
