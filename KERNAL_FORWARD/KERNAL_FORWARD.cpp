#include <stdint.h>     // dùng int8_t, int32_t
#include <hls_math.h>   // dùng half và hls::half_to_float
#include "KERNAL_FORWARD.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_half.h"

/**
 * @brief Kernel HLS cơ bản cho MatMul A(float) * B(Q8_0) -> C(float)
 *
 * @param A     Con trỏ đến Ma trận A (float) [M x K]
 * @param B     Con trỏ đến Ma trận B (Q8_0) [K x N]
 * @param C     Con trỏ đến Ma trận C (float) [M x N]
 * @param M     Số hàng của A
 * @param K     Số cột của A / Số hàng của B
 * @param N     Số cột của B
 */

extern "C" {
void KERNAL_FORWARD(const float *A,
                    const block_q8_0 *B,
                    float *C,
                    int M,
                    int K,
                    int N) {
    // =================== GIAO TIẾP VỚI AXI ===================
#pragma HLS INTERFACE m_axi port=A depth=2048 bundle=gmem0
#pragma HLS INTERFACE m_axi port=B depth=4096 bundle=gmem1
#pragma HLS INTERFACE m_axi port=C depth=2048 bundle=gmem0
#pragma HLS data_pack variable=block_q8_0
  // ================== AXI LITE( CAC THAM SO DIEU KHIEN) ======
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // =================== LOGIC TÍNH TOÁN ===================
row_loop:
    for (int m = 0; m < M; ++m) {
    col_loop:
        for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II=1
            float sum = 0.0f;

        k_loop:
            for (int k = 0; k < K; ++k) {
                // 1. LAY GIA TRI A[m][k]
                float a_val = A[m * K + k];

                // 2. TIM BLOCK CHUA B[k][n]
                int block_row_start = k * (N / QK8_0);
                int block_idx = block_row_start + (n / QK8_0);
                int idx_in_block = n % QK8_0;

                // 3. DOC BLOCK TU DDR
                const block_q8_0 block = B[block_idx];

                // 4. LAY SCALE VA CONVERT SANG FLOAT
                float d =float(block.d);  // CHUYEN HALF SANG FLOAT

                // 5. LAY GIA TRI LUONG TU HOA INT 8
                int8_t b_q = block.qs[idx_in_block];

                // 6. Dequantize
                float b_val = (float)b_q * d;

                // 7. NHAN VA TICH LUY
                sum += a_val * b_val;
            }

            // 8. GHI KET QUA
            C[m * N + n] = sum;
        }
    }
}
}
