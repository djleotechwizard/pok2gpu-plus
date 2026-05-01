#pragma once
/*
 * Minimal test harness.
 *
 * Tests are plain C functions named test_*().
 * Call RUN(test_foo) in main(); the harness prints pass/fail and returns exit code.
 *
 * Example:
 *   static void test_add(void) {
 *       ASSERT_EQ(1 + 1, 2);
 *   }
 *   int main(void) {
 *       RUN(test_add);
 *       return DONE();
 *   }
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int _passed = 0;
static int _failed = 0;

static inline void _run_test(void (*fn)(void), const char *name) {
    /* We capture failures via a simple setjmp-free approach:
     * each test function calls FAIL() which increments _failed and returns.
     * We track the count before and after. */
    int before = _failed;
    fn();
    if (_failed == before) {
        fprintf(stderr, "pass  %s\n", name);
        _passed++;
    }
    /* _failed was already incremented by FAIL() */
}

#define RUN(fn)  _run_test(fn, #fn)

#define DONE() \
    (fprintf(stderr, "\n%d passed, %d failed\n", _passed, _failed), _failed ? 1 : 0)

#define FAIL(msg) \
    do { \
        fprintf(stderr, "FAIL  %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
        _failed++; \
        return; \
    } while (0)

#define ASSERT(expr) \
    do { if (!(expr)) FAIL("assertion failed: " #expr); } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            fprintf(stderr, "FAIL  %s:%d  expected %lld, got %lld\n", \
                    __FILE__, __LINE__, (long long)(b), (long long)(a)); \
            _failed++; \
            return; \
        } \
    } while (0)

#define ASSERT_EQ_HEX(a, b) \
    do { \
        if ((a) != (b)) { \
            fprintf(stderr, "FAIL  %s:%d  expected 0x%llX, got 0x%llX\n", \
                    __FILE__, __LINE__, \
                    (unsigned long long)(b), (unsigned long long)(a)); \
            _failed++; \
            return; \
        } \
    } while (0)
