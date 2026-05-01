#include <stdio.h>
#include <stdlib.h>
#include "gb.h"

/* SM83 CPU — complete implementation
 * All 256 main opcodes + 256 CB-prefixed opcodes */

void cpu_init(CPU *cpu) {
    cpu->A  = 0x01; cpu->F  = 0xB0;
    cpu->B  = 0x00; cpu->C  = 0x13;
    cpu->D  = 0x00; cpu->E  = 0xD8;
    cpu->H  = 0x01; cpu->L  = 0x4D;
    cpu->SP = 0xFFFE;
    cpu->PC = 0x0100;
    cpu->IME         = false;
    cpu->ime_pending = false;
    cpu->halted      = false;
    cpu->halt_bug    = false;
    cpu->cycles      = 0;
}

static void push16(GB *gb, uint16_t v) {
    gb->cpu.SP -= 2;
    memory_write(gb, gb->cpu.SP + 1, (uint8_t)(v >> 8));
    memory_write(gb, gb->cpu.SP,     (uint8_t)(v & 0xFF));
}

static uint16_t pop16(GB *gb) {
    uint8_t lo = memory_read(gb, gb->cpu.SP);
    uint8_t hi = memory_read(gb, gb->cpu.SP + 1);
    gb->cpu.SP += 2;
    return (uint16_t)((hi << 8) | lo);
}

static uint8_t reg_r(GB *gb, int r) {
    switch (r & 7) {
        case 0: return gb->cpu.B;
        case 1: return gb->cpu.C;
        case 2: return gb->cpu.D;
        case 3: return gb->cpu.E;
        case 4: return gb->cpu.H;
        case 5: return gb->cpu.L;
        case 6: return memory_read(gb, cpu_HL(&gb->cpu));
        case 7: return gb->cpu.A;
    }
    return 0;
}

static void reg_w(GB *gb, int r, uint8_t v) {
    switch (r & 7) {
        case 0: gb->cpu.B = v; return;
        case 1: gb->cpu.C = v; return;
        case 2: gb->cpu.D = v; return;
        case 3: gb->cpu.E = v; return;
        case 4: gb->cpu.H = v; return;
        case 5: gb->cpu.L = v; return;
        case 6: memory_write(gb, cpu_HL(&gb->cpu), v); return;
        case 7: gb->cpu.A = v; return;
    }
}

static void alu_add(CPU *c, uint8_t v, bool with_carry) {
    int cy = (with_carry && GET_C(c)) ? 1 : 0;
    int r  = (int)c->A + (int)v + cy;
    SET_H(c, ((c->A & 0xF) + (v & 0xF) + cy) > 0xF);
    SET_C(c, r > 0xFF);
    c->A = (uint8_t)r;
    SET_Z(c, c->A == 0);
    SET_N(c, false);
}

static void alu_sub(CPU *c, uint8_t v, bool with_carry) {
    int cy = (with_carry && GET_C(c)) ? 1 : 0;
    int r  = (int)c->A - (int)v - cy;
    SET_H(c, ((int)(c->A & 0xF) - (int)(v & 0xF) - cy) < 0);
    SET_C(c, r < 0);
    c->A = (uint8_t)r;
    SET_Z(c, c->A == 0);
    SET_N(c, true);
}

static void alu_and(CPU *c, uint8_t v) {
    c->A &= v;
    c->F = FLAG_H;
    SET_Z(c, c->A == 0);
}

static void alu_xor(CPU *c, uint8_t v) {
    c->A ^= v;
    c->F = 0;
    SET_Z(c, c->A == 0);
}

static void alu_or(CPU *c, uint8_t v) {
    c->A |= v;
    c->F = 0;
    SET_Z(c, c->A == 0);
}

static void alu_cp(CPU *c, uint8_t v) {
    int r = (int)c->A - (int)v;
    SET_H(c, ((int)(c->A & 0xF) - (int)(v & 0xF)) < 0);
    SET_C(c, r < 0);
    SET_Z(c, (r & 0xFF) == 0);
    SET_N(c, true);
}

static uint8_t inc8(CPU *c, uint8_t v) {
    SET_H(c, (v & 0xF) == 0xF);
    v++;
    SET_Z(c, v == 0);
    SET_N(c, false);
    return v;
}

static uint8_t dec8(CPU *c, uint8_t v) {
    SET_H(c, (v & 0xF) == 0);
    v--;
    SET_Z(c, v == 0);
    SET_N(c, true);
    return v;
}

static void add_hl16(CPU *c, uint16_t v) {
    uint16_t hl = cpu_HL(c);
    uint32_t r  = (uint32_t)hl + (uint32_t)v;
    SET_H(c, ((hl & 0xFFF) + (v & 0xFFF)) > 0xFFF);
    SET_C(c, r > 0xFFFF);
    SET_N(c, false);
    cpu_set_HL(c, (uint16_t)r);
}

static int exec_cb(GB *gb) {
    CPU    *c  = &gb->cpu;
    uint8_t op = memory_read(gb, c->PC++);
    int     r  = op & 0x07;
    int     b  = (op >> 3) & 0x07;
    uint8_t v  = reg_r(gb, r);
    bool    hl = (r == 6);

    if (op < 0x40) {
        uint8_t carry_out;
        switch (op >> 3) {
            case 0: carry_out = v >> 7; v = (v << 1)|(carry_out); c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 1: carry_out = v & 1;  v = (v >> 1)|(carry_out<<7); c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 2: carry_out = v >> 7; v = (v << 1)|GET_C(c); c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 3: carry_out = v & 1;  v = (v >> 1)|(GET_C(c)<<7); c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 4: carry_out = v >> 7; v = v << 1; c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 5: carry_out = v & 1;  v = (v >> 1)|(v & 0x80); c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            case 6: v = (uint8_t)(((v & 0x0F)<<4)|((v>>4)&0x0F)); c->F=0; SET_Z(c,v==0); break;
            case 7: carry_out = v & 1; v = v >> 1; c->F=0; SET_Z(c,v==0); SET_C(c,carry_out); break;
            default: carry_out = 0; break;
        }
        reg_w(gb, r, v);
        return hl ? 16 : 8;
    }
    if (op < 0x80) {
        SET_Z(c, !(v & (1 << b)));
        SET_N(c, false);
        SET_H(c, true);
        return hl ? 12 : 8;
    }
    if (op < 0xC0) {
        reg_w(gb, r, v & (uint8_t)~(1 << b));
        return hl ? 16 : 8;
    }
    reg_w(gb, r, v | (uint8_t)(1 << b));
    return hl ? 16 : 8;
}

static int service_interrupt(GB *gb) {
    CPU    *c   = &gb->cpu;
    uint8_t pending = gb->mem.io[0x0F] & gb->mem.IE & 0x1F;
    for (int i = 0; i < 5; i++) {
        if (pending & (1 << i)) {
            gb->mem.io[0x0F] &= (uint8_t)~(1 << i);
            c->IME = false;
            push16(gb, c->PC);
            c->PC = (uint16_t)(0x40 + i * 8);
            if (gb->tracer.enabled) {
                gb->pending_trace.interrupt_fired  = true;
                gb->pending_trace.interrupt_vector = (uint8_t)c->PC;
            }
            return 20;
        }
    }
    return 0;
}

int cpu_step(GB *gb) {
    CPU *c = &gb->cpu;

    uint8_t pending = gb->mem.io[0x0F] & gb->mem.IE & 0x1F;

    if (c->halted) {
        if (!pending) return 4;
        c->halted = false;
        if (c->IME) return service_interrupt(gb);
    }

    if (c->IME && pending) return service_interrupt(gb);

    bool ime_this_cycle = c->ime_pending;

    uint16_t pc = c->PC;
    uint8_t  op = memory_read(gb, c->PC++);

    if (c->halt_bug) { c->PC--; c->halt_bug = false; }

    if (gb->tracer.enabled) {
        TraceEntry *te  = &gb->pending_trace;
        te->cycle       = c->cycles;
        te->pc          = pc;
        te->opcode[0]   = op;
        te->opcode_len  = 1;
        te->mem_write_count = 0;
        te->interrupt_fired = false;
    }

    int cycles = 4;

    switch (op) {
    case 0x00: break;
    case 0x10: c->PC++; break;
    case 0x76:
        if (c->IME) {
            c->halted = true;
        } else {
            pending = gb->mem.io[0x0F] & gb->mem.IE & 0x1F;
            if (!pending) c->halted = true;
            else          c->halt_bug = true;
        }
        break;
    case 0xF3: c->IME = false; c->ime_pending = false; break;
    case 0xFB: c->ime_pending = true; break;
    case 0x06: c->B = memory_read(gb, c->PC++); cycles=8; break;
    case 0x0E: c->C = memory_read(gb, c->PC++); cycles=8; break;
    case 0x16: c->D = memory_read(gb, c->PC++); cycles=8; break;
    case 0x1E: c->E = memory_read(gb, c->PC++); cycles=8; break;
    case 0x26: c->H = memory_read(gb, c->PC++); cycles=8; break;
    case 0x2E: c->L = memory_read(gb, c->PC++); cycles=8; break;
    case 0x3E: c->A = memory_read(gb, c->PC++); cycles=8; break;
    case 0x36: memory_write(gb, cpu_HL(c), memory_read(gb, c->PC++)); cycles=12; break;
    case 0x02: memory_write(gb, cpu_BC(c), c->A); cycles=8; break;
    case 0x12: memory_write(gb, cpu_DE(c), c->A); cycles=8; break;
    case 0x0A: c->A = memory_read(gb, cpu_BC(c)); cycles=8; break;
    case 0x1A: c->A = memory_read(gb, cpu_DE(c)); cycles=8; break;
    case 0x22: memory_write(gb, cpu_HL(c), c->A); cpu_set_HL(c, cpu_HL(c)+1); cycles=8; break;
    case 0x2A: c->A = memory_read(gb, cpu_HL(c)); cpu_set_HL(c, cpu_HL(c)+1); cycles=8; break;
    case 0x32: memory_write(gb, cpu_HL(c), c->A); cpu_set_HL(c, cpu_HL(c)-1); cycles=8; break;
    case 0x3A: c->A = memory_read(gb, cpu_HL(c)); cpu_set_HL(c, cpu_HL(c)-1); cycles=8; break;
    case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46: case 0x47:
    case 0x48: case 0x49: case 0x4A: case 0x4B: case 0x4C: case 0x4D: case 0x4E: case 0x4F:
    case 0x50: case 0x51: case 0x52: case 0x53: case 0x54: case 0x55: case 0x56: case 0x57:
    case 0x58: case 0x59: case 0x5A: case 0x5B: case 0x5C: case 0x5D: case 0x5E: case 0x5F:
    case 0x60: case 0x61: case 0x62: case 0x63: case 0x64: case 0x65: case 0x66: case 0x67:
    case 0x68: case 0x69: case 0x6A: case 0x6B: case 0x6C: case 0x6D: case 0x6E: case 0x6F:
    case 0x70: case 0x71: case 0x72: case 0x73: case 0x74: case 0x75:
    case 0x77: case 0x78: case 0x79: case 0x7A: case 0x7B: case 0x7C: case 0x7D: case 0x7E: case 0x7F: {
        int dst = (op >> 3) & 7, src = op & 7;
        reg_w(gb, dst, reg_r(gb, src));
        cycles = (dst == 6 || src == 6) ? 8 : 4;
        break;
    }
    case 0x01: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); cpu_set_BC(c,(uint16_t)((hi<<8)|lo)); cycles=12; break; }
    case 0x11: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); cpu_set_DE(c,(uint16_t)((hi<<8)|lo)); cycles=12; break; }
    case 0x21: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); cpu_set_HL(c,(uint16_t)((hi<<8)|lo)); cycles=12; break; }
    case 0x31: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); c->SP=(uint16_t)((hi<<8)|lo); cycles=12; break; }
    case 0x08: {
        uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++);
        uint16_t addr=(uint16_t)((hi<<8)|lo);
        memory_write(gb,addr,(uint8_t)(c->SP&0xFF));
        memory_write(gb,addr+1,(uint8_t)(c->SP>>8));
        cycles=20; break;
    }
    case 0xF9: c->SP = cpu_HL(c); cycles=8; break;
    case 0xC1: cpu_set_BC(c, pop16(gb)); cycles=12; break;
    case 0xD1: cpu_set_DE(c, pop16(gb)); cycles=12; break;
    case 0xE1: cpu_set_HL(c, pop16(gb)); cycles=12; break;
    case 0xF1: cpu_set_AF(c, pop16(gb)); cycles=12; break;
    case 0xC5: push16(gb, cpu_BC(c)); cycles=16; break;
    case 0xD5: push16(gb, cpu_DE(c)); cycles=16; break;
    case 0xE5: push16(gb, cpu_HL(c)); cycles=16; break;
    case 0xF5: push16(gb, cpu_AF(c)); cycles=16; break;
    case 0xE8: {
        int8_t e = (int8_t)memory_read(gb, c->PC++);
        uint16_t sp = c->SP;
        c->F = 0;
        SET_H(c, ((sp & 0xF) + ((uint8_t)e & 0xF)) > 0xF);
        SET_C(c, ((sp & 0xFF) + (uint8_t)(e)) > 0xFF);
        c->SP = (uint16_t)(sp + e);
        cycles = 16; break;
    }
    case 0xF8: {
        int8_t e = (int8_t)memory_read(gb, c->PC++);
        uint16_t sp = c->SP;
        c->F = 0;
        SET_H(c, ((sp & 0xF) + ((uint8_t)e & 0xF)) > 0xF);
        SET_C(c, ((sp & 0xFF) + (uint8_t)(e)) > 0xFF);
        cpu_set_HL(c, (uint16_t)(sp + e));
        cycles = 12; break;
    }
    case 0x03: cpu_set_BC(c, cpu_BC(c)+1); cycles=8; break;
    case 0x13: cpu_set_DE(c, cpu_DE(c)+1); cycles=8; break;
    case 0x23: cpu_set_HL(c, cpu_HL(c)+1); cycles=8; break;
    case 0x33: c->SP++; cycles=8; break;
    case 0x0B: cpu_set_BC(c, cpu_BC(c)-1); cycles=8; break;
    case 0x1B: cpu_set_DE(c, cpu_DE(c)-1); cycles=8; break;
    case 0x2B: cpu_set_HL(c, cpu_HL(c)-1); cycles=8; break;
    case 0x3B: c->SP--; cycles=8; break;
    case 0x09: add_hl16(c, cpu_BC(c)); cycles=8; break;
    case 0x19: add_hl16(c, cpu_DE(c)); cycles=8; break;
    case 0x29: add_hl16(c, cpu_HL(c)); cycles=8; break;
    case 0x39: add_hl16(c, c->SP);    cycles=8; break;
    case 0x04: c->B = inc8(c,c->B); break;
    case 0x0C: c->C = inc8(c,c->C); break;
    case 0x14: c->D = inc8(c,c->D); break;
    case 0x1C: c->E = inc8(c,c->E); break;
    case 0x24: c->H = inc8(c,c->H); break;
    case 0x2C: c->L = inc8(c,c->L); break;
    case 0x34: { uint8_t v=inc8(c,memory_read(gb,cpu_HL(c))); memory_write(gb,cpu_HL(c),v); cycles=12; break; }
    case 0x3C: c->A = inc8(c,c->A); break;
    case 0x05: c->B = dec8(c,c->B); break;
    case 0x0D: c->C = dec8(c,c->C); break;
    case 0x15: c->D = dec8(c,c->D); break;
    case 0x1D: c->E = dec8(c,c->E); break;
    case 0x25: c->H = dec8(c,c->H); break;
    case 0x2D: c->L = dec8(c,c->L); break;
    case 0x35: { uint8_t v=dec8(c,memory_read(gb,cpu_HL(c))); memory_write(gb,cpu_HL(c),v); cycles=12; break; }
    case 0x3D: c->A = dec8(c,c->A); break;
    case 0x80: case 0x81: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
        alu_add(c, reg_r(gb,op&7), false); cycles=(op&7)==6?8:4; break;
    case 0x88: case 0x89: case 0x8A: case 0x8B: case 0x8C: case 0x8D: case 0x8E: case 0x8F:
        alu_add(c, reg_r(gb,op&7), true);  cycles=(op&7)==6?8:4; break;
    case 0x90: case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97:
        alu_sub(c, reg_r(gb,op&7), false); cycles=(op&7)==6?8:4; break;
    case 0x98: case 0x99: case 0x9A: case 0x9B: case 0x9C: case 0x9D: case 0x9E: case 0x9F:
        alu_sub(c, reg_r(gb,op&7), true);  cycles=(op&7)==6?8:4; break;
    case 0xA0: case 0xA1: case 0xA2: case 0xA3: case 0xA4: case 0xA5: case 0xA6: case 0xA7:
        alu_and(c, reg_r(gb,op&7)); cycles=(op&7)==6?8:4; break;
    case 0xA8: case 0xA9: case 0xAA: case 0xAB: case 0xAC: case 0xAD: case 0xAE: case 0xAF:
        alu_xor(c, reg_r(gb,op&7)); cycles=(op&7)==6?8:4; break;
    case 0xB0: case 0xB1: case 0xB2: case 0xB3: case 0xB4: case 0xB5: case 0xB6: case 0xB7:
        alu_or(c,  reg_r(gb,op&7)); cycles=(op&7)==6?8:4; break;
    case 0xB8: case 0xB9: case 0xBA: case 0xBB: case 0xBC: case 0xBD: case 0xBE: case 0xBF:
        alu_cp(c,  reg_r(gb,op&7)); cycles=(op&7)==6?8:4; break;
    case 0xC6: alu_add(c, memory_read(gb,c->PC++), false); cycles=8; break;
    case 0xCE: alu_add(c, memory_read(gb,c->PC++), true);  cycles=8; break;
    case 0xD6: alu_sub(c, memory_read(gb,c->PC++), false); cycles=8; break;
    case 0xDE: alu_sub(c, memory_read(gb,c->PC++), true);  cycles=8; break;
    case 0xE6: alu_and(c, memory_read(gb,c->PC++)); cycles=8; break;
    case 0xEE: alu_xor(c, memory_read(gb,c->PC++)); cycles=8; break;
    case 0xF6: alu_or(c,  memory_read(gb,c->PC++)); cycles=8; break;
    case 0xFE: alu_cp(c,  memory_read(gb,c->PC++)); cycles=8; break;
    case 0x07: { uint8_t b7=c->A>>7; c->F=0; SET_C(c,b7); c->A=(c->A<<1)|b7; break; }
    case 0x0F: { uint8_t b0=c->A&1; c->F=0; SET_C(c,b0); c->A=(c->A>>1)|(b0<<7); break; }
    case 0x17: { uint8_t b7=c->A>>7; c->A=(c->A<<1)|GET_C(c); c->F=0; SET_C(c,b7); break; }
    case 0x1F: { uint8_t b0=c->A&1; c->A=(c->A>>1)|(GET_C(c)<<7); c->F=0; SET_C(c,b0); break; }
    case 0x27: {
        uint8_t a = c->A;
        if (!GET_N(c)) {
            if (GET_C(c) || a > 0x99) { a += 0x60; SET_C(c, true); }
            if (GET_H(c) || (a & 0x0F) > 0x09) { a += 0x06; }
        } else {
            if (GET_C(c)) { a -= 0x60; }
            if (GET_H(c)) { a -= 0x06; }
        }
        c->A = a;
        SET_Z(c, c->A == 0);
        SET_H(c, false);
        break;
    }
    case 0x2F: c->A = ~c->A; SET_N(c,true); SET_H(c,true); break;
    case 0x37: SET_N(c,false); SET_H(c,false); SET_C(c,true);   break;
    case 0x3F: SET_N(c,false); SET_H(c,false); SET_C(c,!GET_C(c)); break;
    case 0xC3: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); c->PC=(uint16_t)((hi<<8)|lo); cycles=16; break; }
    case 0xE9: c->PC=cpu_HL(c); cycles=4; break;
    case 0xC2: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if (!GET_Z(c)) { c->PC=(uint16_t)((hi<<8)|lo); cycles=16; } else cycles=12; break; }
    case 0xCA: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if ( GET_Z(c)) { c->PC=(uint16_t)((hi<<8)|lo); cycles=16; } else cycles=12; break; }
    case 0xD2: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if (!GET_C(c)) { c->PC=(uint16_t)((hi<<8)|lo); cycles=16; } else cycles=12; break; }
    case 0xDA: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if ( GET_C(c)) { c->PC=(uint16_t)((hi<<8)|lo); cycles=16; } else cycles=12; break; }
    case 0x18: { int8_t e=(int8_t)memory_read(gb,c->PC++); c->PC=(uint16_t)(c->PC+e); cycles=12; break; }
    case 0x20: { int8_t e=(int8_t)memory_read(gb,c->PC++); if (!GET_Z(c)) { c->PC=(uint16_t)(c->PC+e); cycles=12; } else cycles=8; break; }
    case 0x28: { int8_t e=(int8_t)memory_read(gb,c->PC++); if ( GET_Z(c)) { c->PC=(uint16_t)(c->PC+e); cycles=12; } else cycles=8; break; }
    case 0x30: { int8_t e=(int8_t)memory_read(gb,c->PC++); if (!GET_C(c)) { c->PC=(uint16_t)(c->PC+e); cycles=12; } else cycles=8; break; }
    case 0x38: { int8_t e=(int8_t)memory_read(gb,c->PC++); if ( GET_C(c)) { c->PC=(uint16_t)(c->PC+e); cycles=12; } else cycles=8; break; }
    case 0xCD: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); push16(gb, c->PC); c->PC=(uint16_t)((hi<<8)|lo); cycles=24; break; }
    case 0xC4: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if (!GET_Z(c)) { push16(gb,c->PC); c->PC=(uint16_t)((hi<<8)|lo); cycles=24; } else cycles=12; break; }
    case 0xCC: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if ( GET_Z(c)) { push16(gb,c->PC); c->PC=(uint16_t)((hi<<8)|lo); cycles=24; } else cycles=12; break; }
    case 0xD4: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if (!GET_C(c)) { push16(gb,c->PC); c->PC=(uint16_t)((hi<<8)|lo); cycles=24; } else cycles=12; break; }
    case 0xDC: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); if ( GET_C(c)) { push16(gb,c->PC); c->PC=(uint16_t)((hi<<8)|lo); cycles=24; } else cycles=12; break; }
    case 0xC9: c->PC=pop16(gb); cycles=16; break;
    case 0xD9: c->PC=pop16(gb); c->IME=true; cycles=16; break;
    case 0xC0: if (!GET_Z(c)) { c->PC=pop16(gb); cycles=20; } else cycles=8; break;
    case 0xC8: if ( GET_Z(c)) { c->PC=pop16(gb); cycles=20; } else cycles=8; break;
    case 0xD0: if (!GET_C(c)) { c->PC=pop16(gb); cycles=20; } else cycles=8; break;
    case 0xD8: if ( GET_C(c)) { c->PC=pop16(gb); cycles=20; } else cycles=8; break;
    case 0xC7: push16(gb,c->PC); c->PC=0x00; cycles=16; break;
    case 0xCF: push16(gb,c->PC); c->PC=0x08; cycles=16; break;
    case 0xD7: push16(gb,c->PC); c->PC=0x10; cycles=16; break;
    case 0xDF: push16(gb,c->PC); c->PC=0x18; cycles=16; break;
    case 0xE7: push16(gb,c->PC); c->PC=0x20; cycles=16; break;
    case 0xEF: push16(gb,c->PC); c->PC=0x28; cycles=16; break;
    case 0xF7: push16(gb,c->PC); c->PC=0x30; cycles=16; break;
    case 0xFF: push16(gb,c->PC); c->PC=0x38; cycles=16; break;
    case 0xE0: { uint8_t n=memory_read(gb,c->PC++); memory_write(gb,(uint16_t)(0xFF00|n),c->A); cycles=12; break; }
    case 0xF0: { uint8_t n=memory_read(gb,c->PC++); c->A=memory_read(gb,(uint16_t)(0xFF00|n)); cycles=12; break; }
    case 0xE2: memory_write(gb,(uint16_t)(0xFF00|c->C),c->A); cycles=8; break;
    case 0xF2: c->A=memory_read(gb,(uint16_t)(0xFF00|c->C)); cycles=8; break;
    case 0xEA: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); memory_write(gb,(uint16_t)((hi<<8)|lo),c->A); cycles=16; break; }
    case 0xFA: { uint8_t lo=memory_read(gb,c->PC++); uint8_t hi=memory_read(gb,c->PC++); c->A=memory_read(gb,(uint16_t)((hi<<8)|lo)); cycles=16; break; }
    case 0xCB:
        if (gb->tracer.enabled) {
            gb->pending_trace.opcode[1] = memory_read(gb, c->PC);
            gb->pending_trace.opcode_len = 2;
        }
        cycles = exec_cb(gb);
        break;
    default:
        fprintf(stderr, "Unimplemented opcode 0x%02X at PC=0x%04X\n", op, pc);
        break;
    }

    if (ime_this_cycle) {
        c->IME        = true;
        c->ime_pending = false;
    }

    if (gb->tracer.enabled) {
        TraceEntry *te = &gb->pending_trace;
        te->A = c->A; te->F = c->F;
        te->B = c->B; te->C = c->C;
        te->D = c->D; te->E = c->E;
        te->H = c->H; te->L = c->L;
        te->SP       = c->SP;
        te->PC_after = c->PC;
        tracer_emit(&gb->tracer, te);
    }

    c->cycles += (uint64_t)cycles;
    return cycles;
}
