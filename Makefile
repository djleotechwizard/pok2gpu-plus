# =============================================================================
# pok2gpu-plus Makefile
# =============================================================================
#
# Primary targets:
#   make              — build rl_train, playback, trace_gen
#   make rl_train     — GPU training binary
#   make playback     — SDL2 visualizer
#   make trace_gen    — JIT hot-trace compiler
#   make tools        — gb_emu, cfg_tool, ir_tool, trdiff
#   make tests        — build + run all CPU-side tests
#   make bench        — throughput benchmark (requires ROM)
#   make profile      — collect block-hit profile for JIT (requires ROM)
#   make traces       — generate hot_traces.cuh from profile, rebuild rl_train
#   make nsys         — Nsight Systems timeline profile (requires ROM)
#   make ncu          — Nsight Compute deep-dive profile (requires ROM)
#   make clean        — remove all build artefacts
#
# Variables (override on command line):
#   ROM=pokered.gb    ROM path for profiling / training
#   GENS=5            Generations for profile / bench
#   AGENTS=2048       Parallel agents for bench
#   SM=89             CUDA compute capability (89 = RTX 40-series)
#   NVTX=1            Enable NVTX annotations (links libnvToolsExt)
# =============================================================================

# ── Compilers ─────────────────────────────────────────────────────────────────
CC   := gcc
NVCC := /usr/local/cuda/bin/nvcc

# ── Build directory ───────────────────────────────────────────────────────────
BUILD := build

# ── Variables ─────────────────────────────────────────────────────────────────
SM     ?= 89
ROM    ?= pokered.gb
GENS   ?= 5
AGENTS ?= 2048
NVTX   ?= 0
PROFILE_GENS ?= $(GENS)

# ── Include paths ─────────────────────────────────────────────────────────────
INCS := -Iemu -Ianalysis -Itrain -Igpu -I.

# ── C flags (shared host code) ────────────────────────────────────────────────
CFLAGS     := -std=c11 -Wall -Wextra -O2 -g $(INCS)
CFLAGS_DBG := -std=c11 -Wall -Wextra -O0 -g $(INCS)

# ── NVCC flags ────────────────────────────────────────────────────────────────
# -dc: separate device compilation — each .cu produces a relocatable device obj.
# The final link step (nvcc without -dc) merges device code and links the runtime.
NVCC_COMMON := -arch=sm_$(SM) $(INCS) \
               -Xptxas=-v,-warn-lmem-usage \
               -diag-suppress 177 \
               -lineinfo

NVCC_COMPILE := $(NVCC_COMMON) -O3 -dc \
                -Xcompiler -std=c11,-Wall,-Wextra

NVCC_LINK    := $(NVCC_COMMON) -O3

ifeq ($(NVTX),1)
  NVCC_COMPILE += -DPOKE_NVTX
  NVTX_LIBS    := -lnvToolsExt
else
  NVTX_LIBS    :=
endif

# ── SDL2 ──────────────────────────────────────────────────────────────────────
SDL2_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "")
SDL2_LIBS   := $(shell pkg-config --libs   sdl2 2>/dev/null || echo "-lSDL2")

# ── Source → object mappings ──────────────────────────────────────────────────
EMU_SRCS := \
    emu/gb.c      \
    emu/cpu.c     \
    emu/memory.c  \
    emu/ppu.c     \
    emu/timer.c   \
    emu/trace.c

ANALYSIS_SRCS := \
    analysis/disasm.c    \
    analysis/cfg.c       \
    analysis/ir_lift.c   \
    analysis/ir_interp.c

TRAIN_SRCS := train/batch.c

GPU_SRCS := gpu/batch.cu

# Host objects (compiled with gcc)
EMU_OBJS      := $(patsubst %.c,$(BUILD)/%.o,$(EMU_SRCS))
ANALYSIS_OBJS := $(patsubst %.c,$(BUILD)/%.o,$(ANALYSIS_SRCS))
TRAIN_OBJS    := $(patsubst %.c,$(BUILD)/%.o,$(TRAIN_SRCS))

# Device object (compiled with nvcc -dc)
GPU_OBJ       := $(patsubst %.cu,$(BUILD)/%.o,$(GPU_SRCS))

# Training entry-point object
TRAIN_MAIN_OBJ := $(BUILD)/train/train.o

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all rl_train playback trace_gen tools \
        tests test_cpu test_memory test_timer test_ir test_intro \
        bench profile traces nsys ncu clean

# ── Default ───────────────────────────────────────────────────────────────────
all: rl_train playback trace_gen

# ── Directory creation ────────────────────────────────────────────────────────
$(BUILD)/emu $(BUILD)/analysis $(BUILD)/train $(BUILD)/gpu:
	mkdir -p $@

# ── Pattern rules: C → .o ─────────────────────────────────────────────────────
$(BUILD)/emu/%.o: emu/%.c | $(BUILD)/emu
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/analysis/%.o: analysis/%.c | $(BUILD)/analysis
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/train/%.o: train/%.c | $(BUILD)/train
	$(CC) $(CFLAGS) -c $< -o $@

# ── CUDA device object ────────────────────────────────────────────────────────
$(GPU_OBJ): gpu/batch.cu gpu/batch.h gpu/rl.h gpu/perf.cuh gpu/hot_traces.cuh \
            analysis/ir.h analysis/cfg.h emu/gb.h | $(BUILD)/gpu
	$(NVCC) $(NVCC_COMPILE) -c $< -o $@

# ── rl_train ──────────────────────────────────────────────────────────────────
# nvcc links the final binary so it can inject the CUDA device-link step.
rl_train: $(TRAIN_MAIN_OBJ) $(EMU_OBJS) $(ANALYSIS_OBJS) $(TRAIN_OBJS) $(GPU_OBJ)
	$(NVCC) $(NVCC_LINK) -o $@ $^ -lm $(NVTX_LIBS)

# ── playback ──────────────────────────────────────────────────────────────────
playback: apps/playback.c $(EMU_OBJS) emu/gb.h gpu/rl.h
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -o $@ apps/playback.c $(EMU_OBJS) $(SDL2_LIBS)

# ── trace_gen ─────────────────────────────────────────────────────────────────
trace_gen: apps/trace_gen.c \
           $(BUILD)/analysis/disasm.o $(BUILD)/analysis/cfg.o \
           $(BUILD)/analysis/ir_lift.o
	$(CC) $(CFLAGS) -o $@ apps/trace_gen.c \
	    $(BUILD)/analysis/disasm.o $(BUILD)/analysis/cfg.o \
	    $(BUILD)/analysis/ir_lift.o

# ── tools ─────────────────────────────────────────────────────────────────────
tools: gb_emu cfg_tool ir_tool trdiff

gb_emu: apps/gb_emu.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ apps/gb_emu.c $(EMU_SRCS)

cfg_tool: apps/cfg_tool.c analysis/cfg.c analysis/disasm.c
	$(CC) $(CFLAGS_DBG) -o $@ apps/cfg_tool.c analysis/cfg.c analysis/disasm.c

ir_tool: apps/ir_tool.c analysis/cfg.c analysis/disasm.c analysis/ir_lift.c
	$(CC) $(CFLAGS_DBG) -o $@ apps/ir_tool.c \
	    analysis/cfg.c analysis/disasm.c analysis/ir_lift.c

trdiff: apps/trdiff.c
	$(CC) -std=c11 -Wall -Wextra -O2 -o $@ apps/trdiff.c

# ── tests ─────────────────────────────────────────────────────────────────────
tests: test_cpu test_memory test_timer test_ir

test_cpu: tests/test_cpu.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ tests/test_cpu.c $(EMU_SRCS)
	@echo "--- test_cpu ---"; ./$@ || (echo "FAILED: test_cpu"; exit 1)

test_memory: tests/test_memory.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ tests/test_memory.c $(EMU_SRCS)
	@echo "--- test_memory ---"; ./$@ || (echo "FAILED: test_memory"; exit 1)

test_timer: tests/test_timer.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ tests/test_timer.c $(EMU_SRCS)
	@echo "--- test_timer ---"; ./$@ || (echo "FAILED: test_timer"; exit 1)

test_ir: tests/test_ir.c $(EMU_SRCS) $(ANALYSIS_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ tests/test_ir.c $(EMU_SRCS) $(ANALYSIS_SRCS)
	@echo "--- test_ir ---"; ./$@ || (echo "FAILED: test_ir"; exit 1)

test_intro: tests/test_intro.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ tests/test_intro.c $(EMU_SRCS)
	@echo "--- test_intro (ROM=$(ROM)) ---"; ROM=$(ROM) ./$@

# ── JIT hot-trace workflow ────────────────────────────────────────────────────
profile: rl_train
	./rl_train -P $(PROFILE_GENS) -o best_seq.bin $(ROM)
	@echo ""
	@echo "Next: run 'make traces ROM=$(ROM)' to generate JIT hot-traces."

traces: trace_gen
	./trace_gen best_seq.bin.profile $(ROM)
	$(MAKE) $(GPU_OBJ) rl_train
	@echo ""
	@echo "JIT traces active. Re-run 'make bench' to measure the speedup."

# ── Profiling helpers ─────────────────────────────────────────────────────────
bench: rl_train
	ROM=$(ROM) GENS=$(GENS) AGENTS=$(AGENTS) ./profiling/benchmark.sh

nsys: rl_train
	ROM=$(ROM) GENS=$(GENS) AGENTS=$(AGENTS) ./profiling/nsys.sh

ncu: rl_train
	ROM=$(ROM) ./profiling/ncu.sh

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD)/
	rm -f rl_train playback trace_gen \
	      gb_emu cfg_tool ir_tool trdiff \
	      test_cpu test_memory test_timer test_ir test_intro \
	      *.trace *.json *.bin *.profile
	rm -rf profiling/reports/
