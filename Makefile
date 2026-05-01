# =============================================================================
# pok2gpu-plus Makefile
# =============================================================================
#
# Primary targets:
#   make              — build train, playback, trace_gen
#   make train        — GPU training binary
#   make playback     — SDL2 visualizer
#   make trace_gen    — JIT hot-trace compiler
#   make tools        — gb_emu, cfg_tool, ir_tool, trdiff
#   make tests        — build + run all CPU-side tests
#   make bench        — throughput benchmark (requires ROM)
#   make profile      — collect block-hit profile for JIT (requires ROM)
#   make traces       — generate hot_traces.cuh from profile, rebuild train
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
CC      := gcc
NVCC    := /usr/local/cuda/bin/nvcc

# ── Source roots ──────────────────────────────────────────────────────────────
EMU_DIR      := emu
ANALYSIS_DIR := analysis
GPU_DIR      := gpu
TRAIN_DIR    := train
APPS_DIR     := apps
TESTS_DIR    := tests

# ── Compiler flags ────────────────────────────────────────────────────────────
SM          ?= 89
ROM         ?= pokered.gb
GENS        ?= 5
AGENTS      ?= 2048
NVTX        ?= 0

CFLAGS      := -std=c11 -Wall -Wextra -O2 -g \
               -I$(EMU_DIR) -I$(ANALYSIS_DIR) -I$(TRAIN_DIR) \
               -I$(GPU_DIR) -I.

CFLAGS_DBG  := -std=c11 -Wall -Wextra -O0 -g \
               -I$(EMU_DIR) -I$(ANALYSIS_DIR) -I$(TRAIN_DIR) \
               -I$(GPU_DIR) -I.

# NVCC flags: O3, architecture-specific, lineinfo for ncu/nsys, PTX verbosity
NVCC_FLAGS  := -O3 -arch=sm_$(SM) \
               -Iemu -Ianalysis -Itrain -Igpu -I. \
               -Xcompiler -std=c11,-Wall,-Wextra \
               -Xptxas=-v,-warn-lmem-usage \
               -diag-suppress 177 \
               -lineinfo

ifeq ($(NVTX),1)
  NVCC_FLAGS += -DPOKE_NVTX
  NVTX_LIBS  := -lnvToolsExt
else
  NVTX_LIBS  :=
endif

SDL2_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "")
SDL2_LIBS   := $(shell pkg-config --libs   sdl2 2>/dev/null || echo "-lSDL2")

# ── Source file lists ─────────────────────────────────────────────────────────
EMU_SRCS := \
    $(EMU_DIR)/gb.c      \
    $(EMU_DIR)/cpu.c     \
    $(EMU_DIR)/memory.c  \
    $(EMU_DIR)/ppu.c     \
    $(EMU_DIR)/timer.c   \
    $(EMU_DIR)/trace.c

ANALYSIS_SRCS := \
    $(ANALYSIS_DIR)/disasm.c    \
    $(ANALYSIS_DIR)/cfg.c       \
    $(ANALYSIS_DIR)/ir_lift.c   \
    $(ANALYSIS_DIR)/ir_interp.c

TRAIN_SRCS := \
    $(TRAIN_DIR)/batch.c

GPU_SRCS := $(GPU_DIR)/batch.cu

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all train playback trace_gen tools \
        tests test_cpu test_memory test_timer test_ir test_intro \
        bench profile traces nsys ncu clean

# ── Default target ────────────────────────────────────────────────────────────
all: train playback trace_gen

# ── train ─────────────────────────────────────────────────────────────────────
# NVCC compiles everything (C + CUDA) in a single invocation so the linker
# sees both the CUDA runtime and the standard C objects.
train: $(TRAIN_DIR)/train.c \
       $(EMU_SRCS) $(ANALYSIS_SRCS) $(TRAIN_SRCS) $(GPU_SRCS) \
       $(EMU_DIR)/gb.h $(TRAIN_DIR)/batch.h $(GPU_DIR)/batch.h \
       $(GPU_DIR)/rl.h $(GPU_DIR)/hot_traces.cuh $(GPU_DIR)/perf.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ \
	    $(TRAIN_DIR)/train.c \
	    $(EMU_SRCS) $(ANALYSIS_SRCS) $(TRAIN_SRCS) $(GPU_SRCS) \
	    -lm $(NVTX_LIBS)

# ── playback ──────────────────────────────────────────────────────────────────
playback: $(APPS_DIR)/playback.c $(EMU_SRCS) \
          $(EMU_DIR)/gb.h $(GPU_DIR)/rl.h
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -o $@ \
	    $(APPS_DIR)/playback.c $(EMU_SRCS) \
	    $(SDL2_LIBS)

# ── trace_gen ─────────────────────────────────────────────────────────────────
trace_gen: $(APPS_DIR)/trace_gen.c \
           $(ANALYSIS_DIR)/disasm.c $(ANALYSIS_DIR)/cfg.c \
           $(ANALYSIS_DIR)/ir_lift.c
	$(CC) $(CFLAGS) -o $@ \
	    $(APPS_DIR)/trace_gen.c \
	    $(ANALYSIS_DIR)/disasm.c $(ANALYSIS_DIR)/cfg.c \
	    $(ANALYSIS_DIR)/ir_lift.c

# ── debug/analysis tools ──────────────────────────────────────────────────────
tools: gb_emu cfg_tool ir_tool trdiff

gb_emu: $(APPS_DIR)/gb_emu.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(APPS_DIR)/gb_emu.c $(EMU_SRCS)

cfg_tool: $(APPS_DIR)/cfg_tool.c \
          $(ANALYSIS_DIR)/cfg.c $(ANALYSIS_DIR)/disasm.c
	$(CC) $(CFLAGS_DBG) -o $@ \
	    $(APPS_DIR)/cfg_tool.c \
	    $(ANALYSIS_DIR)/cfg.c $(ANALYSIS_DIR)/disasm.c

ir_tool: $(APPS_DIR)/ir_tool.c \
         $(ANALYSIS_DIR)/cfg.c $(ANALYSIS_DIR)/disasm.c \
         $(ANALYSIS_DIR)/ir_lift.c
	$(CC) $(CFLAGS_DBG) -o $@ \
	    $(APPS_DIR)/ir_tool.c \
	    $(ANALYSIS_DIR)/cfg.c $(ANALYSIS_DIR)/disasm.c \
	    $(ANALYSIS_DIR)/ir_lift.c

trdiff: $(APPS_DIR)/trdiff.c
	$(CC) -std=c11 -Wall -Wextra -O2 -o $@ $(APPS_DIR)/trdiff.c

# ── tests ─────────────────────────────────────────────────────────────────────
# Each test is a self-contained binary; 'make tests' builds and runs them all.
tests: test_cpu test_memory test_timer test_ir

test_cpu: $(TESTS_DIR)/test_cpu.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(TESTS_DIR)/test_cpu.c $(EMU_SRCS)
	@echo "--- test_cpu ---"; ./$@ || (echo "FAILED: test_cpu"; exit 1)

test_memory: $(TESTS_DIR)/test_memory.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(TESTS_DIR)/test_memory.c $(EMU_SRCS)
	@echo "--- test_memory ---"; ./$@ || (echo "FAILED: test_memory"; exit 1)

test_timer: $(TESTS_DIR)/test_timer.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(TESTS_DIR)/test_timer.c $(EMU_SRCS)
	@echo "--- test_timer ---"; ./$@ || (echo "FAILED: test_timer"; exit 1)

test_ir: $(TESTS_DIR)/test_ir.c \
         $(EMU_SRCS) $(ANALYSIS_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ \
	    $(TESTS_DIR)/test_ir.c \
	    $(EMU_SRCS) $(ANALYSIS_SRCS)
	@echo "--- test_ir ---"; ./$@ || (echo "FAILED: test_ir"; exit 1)

# Requires ROM — skip if not present
test_intro: $(TESTS_DIR)/test_intro.c $(EMU_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(TESTS_DIR)/test_intro.c $(EMU_SRCS)
	@echo "--- test_intro (ROM=$(ROM)) ---"; ROM=$(ROM) ./$@

# ── JIT hot-trace workflow ────────────────────────────────────────────────────
PROFILE_GENS ?= $(GENS)

# Step 1: run PROFILE_GENS generations and write best_seq.bin.profile
profile: train
	./train -P $(PROFILE_GENS) -o best_seq.bin $(ROM)
	@echo ""
	@echo "Next: run 'make traces ROM=$(ROM)' to generate JIT hot-traces."

# Step 2: compile profile → gpu/hot_traces.cuh, then rebuild train
traces: trace_gen
	./trace_gen best_seq.bin.profile $(ROM)
	$(MAKE) train
	@echo ""
	@echo "JIT traces active. Re-run 'make bench' to measure the speedup."

# ── Profiling helpers ─────────────────────────────────────────────────────────
bench: train
	ROM=$(ROM) GENS=$(GENS) AGENTS=$(AGENTS) ./profiling/benchmark.sh

nsys: train
	ROM=$(ROM) GENS=$(GENS) AGENTS=$(AGENTS) ./profiling/nsys.sh

ncu: train
	ROM=$(ROM) ./profiling/ncu.sh

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f train playback trace_gen \
	      gb_emu cfg_tool ir_tool trdiff \
	      test_cpu test_memory test_timer test_ir test_intro \
	      *.trace *.json *.bin *.profile
	rm -rf profiling/reports/
