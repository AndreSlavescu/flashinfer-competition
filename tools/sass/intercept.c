/*
 * memcpy interceptor for extracting SASS instruction metadata from nvdisasm.
 *
 * nvdisasm embeds instruction descriptions and scheduling latencies as
 * internal strings. This shared library intercepts memcpy calls to capture
 * those strings as they're copied during disassembly.
 *
 * Adapted from: https://github.com/0xD0GF00D/DocumentSASS
 *
 * Build: gcc -shared -fPIC -o intercept.so intercept.c -ldl
 * Usage: LD_PRELOAD=./intercept.so nvdisasm kernel.cubin > raw_output.txt
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <string.h>

void *memcpy(void *__restrict dest, const void *__restrict src, size_t num) {
    void *(*real_memcpy)(void *__restrict, const void *__restrict, size_t) =
        dlsym(RTLD_NEXT, "memcpy");

    /* Print delimiter with dest, src pointers and byte count */
    printf("\n<%p %p %zu>\n", dest, src, num);
    fflush(stdout);

    /* Dump the raw source bytes (contains the instruction metadata) */
    fwrite(src, sizeof(char), num, stdout);
    fflush(stdout);

    return real_memcpy(dest, src, num);
}
