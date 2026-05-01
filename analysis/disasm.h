#pragma once
#include <stdint.h>

/*
 * SM83 disassembler — decodes one instruction and returns its metadata.
 *
 * Control-flow categories used by the CFG builder:
 *
 *   CF_NEXT      normal instruction, continues to pc+len
 *   CF_JUMP      unconditional jump to .target
 *   CF_BRANCH    conditional: taken→.target, not-taken→pc+len
 *   CF_CALL      unconditional call: .target is callee, pc+len is return
 *   CF_CALL_CC   conditional call: same split as CF_BRANCH but via CALL
 *   CF_RET       return (RET / RETI): no successors
 *   CF_RET_CC    conditional return (RET cc): .fallthrough = pc+len
 *   CF_HALT      HALT / STOP: no successors
 *   CF_INDIRECT  JP HL: target unknown statically
 */

typedef enum {
    CF_NEXT,
    CF_JUMP,
    CF_BRANCH,
    CF_CALL,
    CF_CALL_CC,
    CF_RET,
    CF_RET_CC,
    CF_HALT,
    CF_INDIRECT,
} CFType;

typedef struct {
    uint16_t addr;
    uint8_t  bytes[4];
    int      len;          /* instruction length in bytes (1–3) */
    char     mnemonic[48]; /* formatted mnemonic + operands    */
    CFType   cf;
    uint16_t target;       /* branch/jump/call target (if applicable) */
    int      cycles;       /* T-cycles (taken path for branches) */
} Insn;

/*
 * disasm_decode
 *   mem    : 0x8000-byte view of GB address space 0x0000–0x7FFF
 *   pc     : address to decode
 *   out    : filled on success
 *   returns: instruction length in bytes, or 0 on error
 */
int disasm_decode(const uint8_t *mem, uint16_t pc, Insn *out);

/* Human-readable CF type name */
const char *cf_name(CFType cf);
