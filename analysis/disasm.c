#include <stdio.h>
#include <string.h>
#include "disasm.h"

const char *cf_name(CFType cf) {
    switch (cf) {
        case CF_NEXT:     return "NEXT";
        case CF_JUMP:     return "JUMP";
        case CF_BRANCH:   return "BRANCH";
        case CF_CALL:     return "CALL";
        case CF_CALL_CC:  return "CALL_CC";
        case CF_RET:      return "RET";
        case CF_RET_CC:   return "RET_CC";
        case CF_HALT:     return "HALT";
        case CF_INDIRECT: return "INDIRECT";
        default:          return "?";
    }
}

/* ---- Helpers ---- */

/* Read a byte from the view, clamped to 0x7FFF */
static inline uint8_t rb(const uint8_t *m, uint16_t a) { return m[a & 0x7FFF]; }

/* Read 16-bit little-endian */
static inline uint16_t rw(const uint8_t *m, uint16_t a) {
    return (uint16_t)(rb(m, a) | (rb(m, (uint16_t)(a + 1)) << 8));
}

/* Compute relative branch target: PC after instruction + signed offset */
static inline uint16_t rel_target(uint16_t pc, int8_t e) {
    return (uint16_t)((pc + 2) + e); /* pc+2 = address of next instruction */
}

int disasm_decode(const uint8_t *mem, uint16_t pc, Insn *out) {
    memset(out, 0, sizeof(*out));
    out->addr = pc;
    out->cf   = CF_NEXT;

    uint8_t op = rb(mem, pc);
    out->bytes[0] = op;

    /* Default: 1 byte, 4 cycles, normal flow */
    out->len    = 1;
    out->cycles = 4;

#define I1(mn)          do { snprintf(out->mnemonic, 48, "%s", (mn)); out->len=1; } while(0)
#define I2(mn, ...)     do { snprintf(out->mnemonic, 48, (mn), __VA_ARGS__); out->len=2; out->bytes[1]=rb(mem,(uint16_t)(pc+1)); } while(0)
#define I3(mn, ...)     do { snprintf(out->mnemonic, 48, (mn), __VA_ARGS__); out->len=3; out->bytes[1]=rb(mem,(uint16_t)(pc+1)); out->bytes[2]=rb(mem,(uint16_t)(pc+2)); } while(0)

    uint8_t  n8  = rb(mem, (uint16_t)(pc + 1));
    uint16_t n16 = rw(mem, (uint16_t)(pc + 1));
    int8_t   e8  = (int8_t)n8;

    /* Register names for encoding */
    static const char *rn8[8]  = {"B","C","D","E","H","L","[HL]","A"};
    static const char *rn16[4] = {"BC","DE","HL","SP"};
    static const char *rn16af[4] = {"BC","DE","HL","AF"};
    static const char *cc[4]   = {"NZ","Z","NC","C"};

    switch (op) {

    /* ---- 0x00: NOP ---- */
    case 0x00: I1("NOP"); break;

    /* ---- 0x01–0x3F ---- */
    case 0x01: I3("LD BC,$%04X", n16); out->cycles=12; break;
    case 0x02: I1("LD [BC],A"); out->cycles=8; break;
    case 0x03: I1("INC BC"); out->cycles=8; break;
    case 0x04: I1("INC B"); break;
    case 0x05: I1("DEC B"); break;
    case 0x06: I2("LD B,$%02X", n8); out->cycles=8; break;
    case 0x07: I1("RLCA"); break;
    case 0x08: I3("LD [$%04X],SP", n16); out->cycles=20; break;
    case 0x09: I1("ADD HL,BC"); out->cycles=8; break;
    case 0x0A: I1("LD A,[BC]"); out->cycles=8; break;
    case 0x0B: I1("DEC BC"); out->cycles=8; break;
    case 0x0C: I1("INC C"); break;
    case 0x0D: I1("DEC C"); break;
    case 0x0E: I2("LD C,$%02X", n8); out->cycles=8; break;
    case 0x0F: I1("RRCA"); break;

    case 0x10: I2("STOP $%02X", n8); out->cf=CF_HALT; out->cycles=4; break;
    case 0x11: I3("LD DE,$%04X", n16); out->cycles=12; break;
    case 0x12: I1("LD [DE],A"); out->cycles=8; break;
    case 0x13: I1("INC DE"); out->cycles=8; break;
    case 0x14: I1("INC D"); break;
    case 0x15: I1("DEC D"); break;
    case 0x16: I2("LD D,$%02X", n8); out->cycles=8; break;
    case 0x17: I1("RLA"); break;
    case 0x18: /* JR e */
        I2("JR %+d", e8);
        out->cf     = CF_JUMP;
        out->target = rel_target(pc, e8);
        out->cycles = 12;
        break;
    case 0x19: I1("ADD HL,DE"); out->cycles=8; break;
    case 0x1A: I1("LD A,[DE]"); out->cycles=8; break;
    case 0x1B: I1("DEC DE"); out->cycles=8; break;
    case 0x1C: I1("INC E"); break;
    case 0x1D: I1("DEC E"); break;
    case 0x1E: I2("LD E,$%02X", n8); out->cycles=8; break;
    case 0x1F: I1("RRA"); break;

    case 0x20: /* JR NZ, e */
        I2("JR NZ,%+d", e8);
        out->cf=CF_BRANCH; out->target=rel_target(pc,e8); out->cycles=12; break;
    case 0x21: I3("LD HL,$%04X", n16); out->cycles=12; break;
    case 0x22: I1("LD [HL+],A"); out->cycles=8; break;
    case 0x23: I1("INC HL"); out->cycles=8; break;
    case 0x24: I1("INC H"); break;
    case 0x25: I1("DEC H"); break;
    case 0x26: I2("LD H,$%02X", n8); out->cycles=8; break;
    case 0x27: I1("DAA"); break;
    case 0x28: /* JR Z, e */
        I2("JR Z,%+d", e8);
        out->cf=CF_BRANCH; out->target=rel_target(pc,e8); out->cycles=12; break;
    case 0x29: I1("ADD HL,HL"); out->cycles=8; break;
    case 0x2A: I1("LD A,[HL+]"); out->cycles=8; break;
    case 0x2B: I1("DEC HL"); out->cycles=8; break;
    case 0x2C: I1("INC L"); break;
    case 0x2D: I1("DEC L"); break;
    case 0x2E: I2("LD L,$%02X", n8); out->cycles=8; break;
    case 0x2F: I1("CPL"); break;

    case 0x30: /* JR NC, e */
        I2("JR NC,%+d", e8);
        out->cf=CF_BRANCH; out->target=rel_target(pc,e8); out->cycles=12; break;
    case 0x31: I3("LD SP,$%04X", n16); out->cycles=12; break;
    case 0x32: I1("LD [HL-],A"); out->cycles=8; break;
    case 0x33: I1("INC SP"); out->cycles=8; break;
    case 0x34: I1("INC [HL]"); out->cycles=12; break;
    case 0x35: I1("DEC [HL]"); out->cycles=12; break;
    case 0x36: I2("LD [HL],$%02X", n8); out->cycles=12; break;
    case 0x37: I1("SCF"); break;
    case 0x38: /* JR C, e */
        I2("JR C,%+d", e8);
        out->cf=CF_BRANCH; out->target=rel_target(pc,e8); out->cycles=12; break;
    case 0x39: I1("ADD HL,SP"); out->cycles=8; break;
    case 0x3A: I1("LD A,[HL-]"); out->cycles=8; break;
    case 0x3B: I1("DEC SP"); out->cycles=8; break;
    case 0x3C: I1("INC A"); break;
    case 0x3D: I1("DEC A"); break;
    case 0x3E: I2("LD A,$%02X", n8); out->cycles=8; break;
    case 0x3F: I1("CCF"); break;

    /* ---- LD r,r / HALT (0x40–0x7F) ---- */
    case 0x76:
        I1("HALT");
        out->cf     = CF_HALT;
        out->cycles = 4;
        break;

    default:
        if (op >= 0x40 && op <= 0x7F) {
            snprintf(out->mnemonic, 48, "LD %s,%s",
                     rn8[(op>>3)&7], rn8[op&7]);
            out->cycles = ((op&7)==6 || ((op>>3)&7)==6) ? 8 : 4;
        }
        /* ALU A,r (0x80–0xBF) */
        else if (op >= 0x80 && op <= 0xBF) {
            static const char *alu_ops[8] = {
                "ADD","ADC","SUB","SBC","AND","XOR","OR","CP"};
            int grp = (op>>3)&7, reg = op&7;
            snprintf(out->mnemonic, 48, "%s A,%s", alu_ops[grp], rn8[reg]);
            out->cycles = (reg==6) ? 8 : 4;
        }
        else {
            snprintf(out->mnemonic, 48, "DB $%02X", op);
        }
        break;

    /* ---- 0x80–0xBF: explicit for clarity ---- */
    /* (Handled in default above, kept separate below for ALU immediates) */

    /* ---- 0xC0–0xFF ---- */
    case 0xC0: /* RET NZ */
        I1("RET NZ"); out->cf=CF_RET_CC; out->cycles=20; break;
    case 0xC1: I1("POP BC"); out->cycles=12; break;
    case 0xC2: /* JP NZ, nn */
        I3("JP NZ,$%04X", n16);
        out->cf=CF_BRANCH; out->target=n16; out->cycles=16; break;
    case 0xC3: /* JP nn */
        I3("JP $%04X", n16);
        out->cf=CF_JUMP; out->target=n16; out->cycles=16; break;
    case 0xC4: /* CALL NZ, nn */
        I3("CALL NZ,$%04X", n16);
        out->cf=CF_CALL_CC; out->target=n16; out->cycles=24; break;
    case 0xC5: I1("PUSH BC"); out->cycles=16; break;
    case 0xC6: I2("ADD A,$%02X", n8); out->cycles=8; break;
    case 0xC7: /* RST 00H */
        I1("RST $00"); out->cf=CF_CALL; out->target=0x0000; out->cycles=16; break;
    case 0xC8: I1("RET Z");  out->cf=CF_RET_CC; out->cycles=20; break;
    case 0xC9: I1("RET");    out->cf=CF_RET;    out->cycles=16; break;
    case 0xCA: /* JP Z, nn */
        I3("JP Z,$%04X", n16);
        out->cf=CF_BRANCH; out->target=n16; out->cycles=16; break;

    case 0xCB: { /* CB prefix */
        uint8_t cb = rb(mem, (uint16_t)(pc+1));
        out->bytes[1] = cb;
        out->len    = 2;
        out->cycles = ((cb&7)==6) ? 16 : 8;
        if (cb < 0x40) {
            /* BIT 12: 12 cycles for (HL), actually for BIT it's different */
            if (cb >= 0x40 && cb < 0x80 && (cb&7)==6) out->cycles=12;
            static const char *cb_ops[8] = {
                "RLC","RRC","RL","RR","SLA","SRA","SWAP","SRL"};
            snprintf(out->mnemonic, 48, "%s %s", cb_ops[cb>>3], rn8[cb&7]);
        } else if (cb < 0x80) {
            if ((cb&7)==6) out->cycles=12;
            snprintf(out->mnemonic, 48, "BIT %d,%s", (cb>>3)&7, rn8[cb&7]);
        } else if (cb < 0xC0) {
            snprintf(out->mnemonic, 48, "RES %d,%s", (cb>>3)&7, rn8[cb&7]);
        } else {
            snprintf(out->mnemonic, 48, "SET %d,%s", (cb>>3)&7, rn8[cb&7]);
        }
        break;
    }

    case 0xCC: /* CALL Z, nn */
        I3("CALL Z,$%04X", n16);
        out->cf=CF_CALL_CC; out->target=n16; out->cycles=24; break;
    case 0xCD: /* CALL nn */
        I3("CALL $%04X", n16);
        out->cf=CF_CALL; out->target=n16; out->cycles=24; break;
    case 0xCE: I2("ADC A,$%02X", n8); out->cycles=8; break;
    case 0xCF: I1("RST $08"); out->cf=CF_CALL; out->target=0x0008; out->cycles=16; break;

    case 0xD0: I1("RET NC"); out->cf=CF_RET_CC; out->cycles=20; break;
    case 0xD1: I1("POP DE"); out->cycles=12; break;
    case 0xD2: I3("JP NC,$%04X", n16); out->cf=CF_BRANCH; out->target=n16; out->cycles=16; break;
    case 0xD3: I1("DB $D3"); break; /* illegal */
    case 0xD4: I3("CALL NC,$%04X", n16); out->cf=CF_CALL_CC; out->target=n16; out->cycles=24; break;
    case 0xD5: I1("PUSH DE"); out->cycles=16; break;
    case 0xD6: I2("SUB A,$%02X", n8); out->cycles=8; break;
    case 0xD7: I1("RST $10"); out->cf=CF_CALL; out->target=0x0010; out->cycles=16; break;
    case 0xD8: I1("RET C");  out->cf=CF_RET_CC; out->cycles=20; break;
    case 0xD9: I1("RETI");   out->cf=CF_RET;    out->cycles=16; break;
    case 0xDA: I3("JP C,$%04X", n16); out->cf=CF_BRANCH; out->target=n16; out->cycles=16; break;
    case 0xDB: I1("DB $DB"); break;
    case 0xDC: I3("CALL C,$%04X", n16); out->cf=CF_CALL_CC; out->target=n16; out->cycles=24; break;
    case 0xDD: I1("DB $DD"); break;
    case 0xDE: I2("SBC A,$%02X", n8); out->cycles=8; break;
    case 0xDF: I1("RST $18"); out->cf=CF_CALL; out->target=0x0018; out->cycles=16; break;

    case 0xE0: I2("LDH [$FF%02X],A", n8); out->cycles=12; break;
    case 0xE1: I1("POP HL"); out->cycles=12; break;
    case 0xE2: I1("LDH [C],A"); out->cycles=8; break;
    case 0xE3: I1("DB $E3"); break;
    case 0xE4: I1("DB $E4"); break;
    case 0xE5: I1("PUSH HL"); out->cycles=16; break;
    case 0xE6: I2("AND A,$%02X", n8); out->cycles=8; break;
    case 0xE7: I1("RST $20"); out->cf=CF_CALL; out->target=0x0020; out->cycles=16; break;
    case 0xE8: I2("ADD SP,%+d", e8); out->cycles=16; break;
    case 0xE9: /* JP HL */
        I1("JP HL"); out->cf=CF_INDIRECT; out->cycles=4; break;
    case 0xEA: I3("LD [$%04X],A", n16); out->cycles=16; break;
    case 0xEB: I1("DB $EB"); break;
    case 0xEC: I1("DB $EC"); break;
    case 0xED: I1("DB $ED"); break;
    case 0xEE: I2("XOR A,$%02X", n8); out->cycles=8; break;
    case 0xEF: I1("RST $28"); out->cf=CF_CALL; out->target=0x0028; out->cycles=16; break;

    case 0xF0: I2("LDH A,[$FF%02X]", n8); out->cycles=12; break;
    case 0xF1: I1("POP AF"); out->cycles=12; break;
    case 0xF2: I1("LDH A,[C]"); out->cycles=8; break;
    case 0xF3: I1("DI"); break;
    case 0xF4: I1("DB $F4"); break;
    case 0xF5: I1("PUSH AF"); out->cycles=16; break;
    case 0xF6: I2("OR A,$%02X", n8); out->cycles=8; break;
    case 0xF7: I1("RST $30"); out->cf=CF_CALL; out->target=0x0030; out->cycles=16; break;
    case 0xF8: I2("LD HL,SP%+d", e8); out->cycles=12; break;
    case 0xF9: I1("LD SP,HL"); out->cycles=8; break;
    case 0xFA: I3("LD A,[$%04X]", n16); out->cycles=16; break;
    case 0xFB: I1("EI"); break;
    case 0xFC: I1("DB $FC"); break;
    case 0xFD: I1("DB $FD"); break;
    case 0xFE: I2("CP A,$%02X", n8); out->cycles=8; break;
    case 0xFF: I1("RST $38"); out->cf=CF_CALL; out->target=0x0038; out->cycles=16; break;

    } /* end switch */

    /* Suppress 'unused' warnings for the static arrays */
    (void)rn16; (void)rn16af; (void)cc;

#undef I1
#undef I2
#undef I3

    return out->len;
}
