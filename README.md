# pok2gpu-plus

Run thousands of Pokémon Red instances in parallel on your GPU to search for
action sequences that explore as many map tiles as possible — using an
evolutionary strategy, no gradients required.

At 2048 parallel agents on an RTX 40-series card it runs at ~90,000
simulated Game Boy frames per second.

---

## How it works

Each GPU thread owns one complete Game Boy instance.  A population of agents
each plays through an episode (e.g. 600 frames) using their own action
sequence.  After each generation the best sequences are kept, then
crossover and mutation generate the next population.  The fitness signal is
simply the number of unique `(map, y, x)` tiles visited during the episode.

The emulator is written from scratch in CUDA C:

- **SM83 CPU** — all 512 opcodes, exact cycle timing, HALT fast-forward
- **PPU** — mode/interrupt timing only (no rendering during training)
- **Timer** — O(1) analytical falling-edge counting
- **MBC1/3/5** — full ROM/RAM banking
- **IR interpreter** — ROM is statically disassembled into a compact
  intermediate representation; the GPU executes IR blocks instead of
  individual opcodes, amortising dispatch overhead
- **Sorted dispatch** — states are sorted by current block key before each
  kernel launch to minimise warp divergence

---

## Requirements

| Dependency | Notes |
|------------|-------|
| NVIDIA GPU | Turing (sm_75) or newer recommended |
| CUDA Toolkit | 11.x or newer (`nvcc`, `cub/cub.cuh`) |
| GCC | C11 support |
| SDL2 | Only needed for the `playback` visualiser |
| Pokémon Red ROM | `pokered.gb`, MD5 `3d45c1ee9abd5738df46d2bdda8b57dc` |

The ROM is **not** included.  Obtain it legally and place it in the project
root as `pokered.gb`.

---

## Quick start

```bash
# 1. Build everything
make all        # rl_train, playback, trace_gen
make tools      # gb_emu, cfg_tool, ir_tool, trdiff  (optional)

# 2. Run the test suite
make tests

# 3. Train — 500 generations, 2048 agents, 600 frames/episode
./rl_train pokered.gb

# 4. Watch the best sequence
./playback pokered.gb best_seq.bin
```

---

## Training

```
./rl_train [options] <rom.gb>

  -n N    Parallel agents          (default 2048)
  -g G    Generations              (default 500)
  -f F    GB frames per episode    (default 600)
  -r R    Action repeat in frames  (default 8 — one tile per step)
  -w W    Scripted intro frames    (default 3600; 0 = skip intro)
  -e E    Elite survivors          (default 64)
  -m M    Mutation rate 0..1       (default 0.05)
  -o FILE Output sequence file     (default best_seq.bin)
  -s SEED Random seed              (default time)
  -H F    Successive Halving: cull bottom 50% after F frames
  -V N    CPU verification every N gens (slow but catches GPU bugs)
  -h      Help
```

### Example: long training run

```bash
./rl_train -n 2048 -g 1000 -f 1800 -r 8 pokered.gb
```

### Example: quick experiment

```bash
./rl_train -n 2048 -g 50 -f 600 pokered.gb
```

### Live progress output

```
gen   42/500  max= 87  mean=61.3  p75= 73  p25= 51  min= 12  ep=12.4s  91k sf/s  mut=0.05  elapsed=9m  eta=1h12m
```

- **max/mean/p75/p25/min** — tile-count distribution across agents
- **sf/s** — state-frames per second (GPU throughput)
- **mut** — current mutation rate (rises automatically after stagnation)
- A line marked `*** NEW BEST ***` means `best_seq.bin` was just updated

---

## Playback

```
./playback [options] <rom.gb> <best_seq.bin>

  -s N   Window scale factor  (default 3 → 480×432 window)
  -x N   Speed multiplier     (default 1 = real GB speed; 0 = unlimited)
  -l N   Loop N times         (0 = loop forever, default)
  -k     Skip intro, jump straight to the RL portion
  -v     Verbose: print tile hits per step
  -h     Help
```

At the end of each loop it prints:

```
loop 1  cpu=87  gpu=87  diff=+0
```

The CPU replay score should match the GPU training score.  A non-zero diff
indicates a divergence bug.

---

## JIT hot-trace acceleration

After initial training the profiler can identify which blocks account for
the vast majority of execution and compile them to straight-line CUDA
device functions, bypassing the IR interpreter entirely.

```bash
# Step 1: collect block-hit profile (runs 5 training generations)
make profile ROM=pokered.gb

# Step 2: generate gpu/hot_traces.cuh and rebuild
make traces ROM=pokered.gb

# Step 3: benchmark the speedup
make bench ROM=pokered.gb
```

Typical result: 5–15% additional throughput on top of the baseline,
depending on which blocks become hot for your specific ROM version and
training configuration.

To revert to the stub (no JIT), run:

```bash
make clean && make all ROM=pokered.gb
```

---

## Benchmarking

```bash
make bench ROM=pokered.gb              # default: 5 gens, 2048 agents
GENS=10 AGENTS=512 make bench ROM=pokered.gb

# Compare two code states
git stash && make all && make bench ROM=pokered.gb > before.txt
git stash pop && make all && make bench ROM=pokered.gb > after.txt
diff before.txt after.txt
```

---

## Profiling

### Nsight Systems (timeline)

```bash
make nsys ROM=pokered.gb
# output: profiling/reports/nsys_<timestamp>.nsys-rep
# open in Nsight Systems GUI
```

### Nsight Compute (kernel metrics)

```bash
# May require elevated permissions on Linux — see note below
make ncu ROM=pokered.gb
# output: profiling/reports/ncu_<timestamp>.ncu-rep
```

**Linux permissions for hardware counters:**

```bash
# Temporary (until reboot):
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Persistent:
echo 'kernel.perf_event_paranoid=0' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl -p /etc/sysctl.d/99-perf.conf
```

---

## Adapting to your GPU

The only thing you need to change for a different GPU is the compute
capability flag.  Pass `SM=` to every `make` invocation:

| GPU generation | `SM=` value |
|----------------|------------|
| Turing (RTX 20xx, GTX 16xx) | `SM=75` |
| Ampere (RTX 30xx, A-series) | `SM=80` or `SM=86` |
| Ada Lovelace (RTX 40xx) | `SM=89` |
| Hopper (H100) | `SM=90` |
| Blackwell (RTX 50xx) | `SM=100` |

```bash
make all SM=86 && make bench ROM=pokered.gb SM=86
```

You can also set it permanently by editing the top of the Makefile:

```makefile
SM ?= 86   # change this line
```

### Tuning agent count for your VRAM

Rule of thumb: each agent uses roughly **50 KB** of GPU memory
(GBStateGPU ~700 B + 8 KB VRAM + 8 KB WRAM + 2 KB tile hash + overhead).

| Agents | Approx VRAM |
|--------|-------------|
| 512    | ~40 MB |
| 1024   | ~80 MB |
| 2048   | ~160 MB |
| 4096   | ~320 MB |
| 8192   | ~640 MB |

On a 4 GB card, 2048 agents is comfortable.  On 8 GB+, try 4096 or 8192:

```bash
./rl_train -n 4096 -g 500 pokered.gb
```

Throughput scales near-linearly up to the point where the GPU is fully
saturated.

### Tuning thread block size

The default is 128 threads per block, which fits well on most Ampere/Ada
cards.  If you see low occupancy with `ncu`, try 64 or 256 by editing
`gpu/batch.cu`:

```c
int threads = 128;  // try 64 or 256
```

---

## Project structure

```
pok2gpu-plus/
├── emu/          SM83 CPU, PPU, timer, memory map, trace logger
├── analysis/     Static disassembler, CFG builder, IR lifter, IR interpreter
├── gpu/          CUDA batch emulator, RL episode kernel, JIT hot-traces
├── train/        CPU batch, evolutionary training loop
├── apps/         Standalone binaries (playback, gb_emu, cfg_tool, ir_tool, trace_gen, trdiff)
├── tests/        CPU-side test suite (33 tests)
├── profiling/    nsys/ncu wrapper scripts, throughput benchmark
└── Makefile
```

### Analysis tools

```bash
# Standalone reference emulator (useful for generating traces)
./gb_emu pokered.gb -f 600 -t out.trace

# Dump CFG as JSON
./cfg_tool pokered.gb -o cfg.json

# Dump IR for a specific bank/address
./ir_tool pokered.gb -b 1 -a 0x4B6C

# Compare two execution traces to find the first divergence
./trdiff ref.trace test.trace
```

---

## Reproducing the optimisations

Two key GPU optimisations are already applied and documented in the source:

**1. `__noinline__` on `gpu_exec_block`** (`gpu/batch.cu`)  
The IR block executor was being inlined into the episode kernel, contributing
528 bytes of local memory per thread and polluting the L1 cache.  Marking it
`__noinline__` drops the kernel frame to 32 bytes, freeing L1 for WRAM reads
and hash-table lookups.  Result: +10% throughput.

**2. Dead flag elimination** (`analysis/ir_lift.c`)  
The IR lifter eagerly emits Z/N/H/C flag updates after every ALU instruction.
In compute-heavy blocks (e.g. the OAM sprite-position loop that accounts for
~10% of all executions) most of these updates are immediately overwritten
before any branch reads them.  A backward liveness pass at IR compile time
converts dead flag writes to IR_NOP and compacts them out, reducing IR op
count by ~49% for the hottest blocks.  Result: +14% throughput.

Combined: **+25% throughput** over the starting baseline (71k → 89k sf/s on
RTX 40-series at 2048 agents).

---

## License

Source code: MIT.  
The Pokémon Red ROM is © Nintendo / Game Freak and is not included.
