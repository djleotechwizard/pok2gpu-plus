#pragma once
#include <stdint.h>
#include <stdio.h>
#include "cfg.h"

/*
 * Intermediate Representation
 *
 * Each SM83 instruction is lifted into a flat list of IR instructions that
 * make every register read/write, flag update, and memory access explicit.
 *
 * Value encoding:
 *   8-bit registers : A B C D E H L
 *   16-bit pairs    : AF BC DE HL SP
 *   Flags           : ZF NF HF CF
 *   Immediates      : IMM8  IMM16
 *   Temporaries     : TMP(n)  — block-local scratch values
 */

typedef enum {
    IRV_NONE = 0,
    IRV_A, IRV_B, IRV_C, IRV_D, IRV_E, IRV_H, IRV_L,
    IRV_AF, IRV_BC, IRV_DE, IRV_HL, IRV_SP,
    IRV_ZF, IRV_NF, IRV_HF, IRV_CF,
    IRV_IMM8,
    IRV_IMM16,
    IRV_TMP,
} IRVKind;

typedef struct {
    IRVKind  kind;
    union {
        uint8_t  u8;
        uint16_t u16;
        int      idx;
    };
} IRVal;

typedef enum {
    IR_NOP = 0,
    IR_MOV,
    IR_LOAD8,
    IR_STORE8,
    IR_ADD8, IR_ADC8, IR_SUB8, IR_SBC8,
    IR_AND8, IR_OR8,  IR_XOR8, IR_NOT8,
    IR_INC8, IR_DEC8,
    IR_ADD16, IR_INC16, IR_DEC16, IR_ADD16_SP,
    IR_SHL8, IR_SHR8, IR_SAR8,
    IR_RLC8, IR_RRC8, IR_RL8, IR_RR8,
    IR_SWAP8, IR_BIT8, IR_RES8, IR_SET8,
    IR_ZFLAG,
    IR_HFLAG_ADD, IR_HFLAG_SUB,
    IR_CFLAG_ADD, IR_CFLAG_SUB,
    IR_HFLAG_A16, IR_CFLAG_A16,
    IR_HFLAG_SP,  IR_CFLAG_SP,
    IR_PUSH16, IR_POP16,
    IR_JUMP, IR_BRANCH, IR_CALL, IR_RET, IR_HALT,
    IR_DAA, IR_IME_SET, IR_IME_CLR,
} IROp;

typedef struct {
    IROp  op;
    IRVal dst;
    IRVal src[3];
} IRInsn;

typedef struct {
    uint16_t entry;
    int      bank;
    ExitType exit_type;
    uint16_t succs[2];
    int      n_succs;

    IRInsn  *insns;
    int      n_insns;
    int      n_src_insns;
    int      cap;
    int      n_tmps;

    int      taken_cycles;
    int      fall_cycles;
} IRBlock;

static inline IRVal irv_none(void)          { return (IRVal){IRV_NONE,  {.u8=0}}; }
static inline IRVal irv_reg(IRVKind k)      { return (IRVal){k,         {.u8=0}}; }
static inline IRVal irv_imm8(uint8_t v)    { IRVal x; x.kind=IRV_IMM8;  x.u8=v;   return x; }
static inline IRVal irv_imm16(uint16_t v)  { IRVal x; x.kind=IRV_IMM16; x.u16=v;  return x; }
static inline IRVal irv_tmp(int i)          { IRVal x; x.kind=IRV_TMP;   x.idx=i;  return x; }

IRBlock    *ir_lift_block(const Block *b);
void        ir_free_block(IRBlock *ib);
void        ir_print_block(const IRBlock *ib, const Block *src, FILE *f);
const char *irop_name(IROp op);
const char *irv_str(const IRVal *v, char *buf, int sz);
