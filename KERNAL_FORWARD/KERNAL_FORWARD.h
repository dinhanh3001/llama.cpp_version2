#pragma once

#include <stdint.h>
#include <hls_math.h>

#define QK8_0 32

typedef half float16_t;

struct block_q8_0 {
    float16_t d;
    int8_t qs[QK8_0];
};

extern "C" {
void KERNAL_FORWARD(
    const float* A,
    const block_q8_0* B,
    float* C,
    int M,
    int K,
    int N
);
} // extern "C"
