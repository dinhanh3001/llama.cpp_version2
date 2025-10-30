#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h> // Cho rand()


#include "KERNAL_FORWARD.h"

// Hàm quantize_row_q8_0 (phiên bản CPU) để TẠO DỮ LIỆU TEST cho B

void quantize_row_q8_0_cpu(const float* x, block_q8_0* y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i * QK8_0 + j];
            if (fabsf(v) > amax) {
                amax = fabsf(v);
            }
        }

        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        y[i].d = d; //

        for (int j = 0; j < QK8_0; ++j) {
            y[i].qs[j] = (int8_t)roundf(x[i * QK8_0 + j] * id);
        }
    }
}

// Hàm tính MatMul trên CPU
// Hàm này phải mô phỏng CHÍNH XÁC logic của kernel HLS
void matmul_cpu_golden(
    const float* A,
    const block_q8_0* B,
    float* C_golden,
    int M, int K, int N
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Lấy A
                float a_val = A[m * K + k];


                int block_row_start = k * (N / QK8_0);
                int block_idx = block_row_start + (n / QK8_0);
                int idx_in_block = n % QK8_0;

                const block_q8_0 block = B[block_idx];
                const float d = block.d;
                const int8_t b_q = block.qs[idx_in_block];
                float b_val = (float)b_q * d;

                // Tích lũy
                sum += a_val * b_val;
            }
            C_golden[m * N + n] = sum;
        }
    }
}

void print_matrix(const char* name, const float* matrix, int M, int N) {
    std::cout << "--- Ma tran: " << name << " (Kich thuoc " << M << "x" << N << ") ---" << std::endl;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            // Dùng printf để căn chỉnh cho đẹp
            printf("%9.4f ", matrix[m * N + n]);
        }
        printf("\n");
    }
    std::cout << "---------------------------------------" << std::endl;
}

void KERNAL_FORWARD(const float * A,
		               const block_q8_0 *B,
					   float * C,
					   int M,
					   int K,
					   int N);

int main() {
    // Kích thước ma trận
    const int M = 4;
    const int K = 64; // PHAI CHIA HET CHO  QK8_0 (32)
    const int N = 64;  // PHAI CHIA HET CHO QK8_0 (32)

    // CAP PHAT BO NHO
    std::vector<float> A_host(M * K);
    // Ma trận B[K][N] được quantize thành K * (N/32) blocks

    std::vector<block_q8_0> B_host(K * N / QK8_0);
    std::vector<float> C_fpga(M * N);
    std::vector<float> C_golden(M * N);

    // TAO DU LIEU FLOAT TAM THOI CHO  B
    std::vector<float> B_float_temp(K * N);

    // 1. KHOI TAO DU LIEU NGAU NHIEN
    srand(123); //CO DINH DU LIEU CO THE TAI LAP
    for (int i = 0; i < M * K; ++i) A_host[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) B_float_temp[i] = (float)rand() / (float)RAND_MAX - 0.5f;


    for (int k_row = 0; k_row < K; ++k_row) {
        quantize_row_q8_0_cpu(
            &B_float_temp[k_row * N],
            &B_host[k_row * (N / QK8_0)],
            N
        );
    }

  // GOI HAM TINH TOAN TREN CPU
    std::cout << "Calculating MATRIX COMPARE result on CPU..." << std::endl;
   matmul_cpu_golden(A_host.data(), B_host.data(), C_golden.data(), M, K, N);

    // 4. GOI HAM TINH TOAN TREN KERNAL
    std::cout << "Calling HLS kernel 'KERNAL_FORWARD'..." << std::endl;
    KERNAL_FORWARD(A_host.data(), B_host.data(), C_fpga.data(), M, K, N);



    // =============== HIEN THI KET QUA MA TRAN =========================
    print_matrix("Ket qua Kernel (FPGA)", C_fpga.data(), M, N);


    print_matrix("Ket qua MATRIX COMPARE (CPU)", C_golden.data(), M, N);
    // ===================================================================

    // 5. SO SANH KET QUA
    std::cout << "Comparing results..." << std::endl;
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_golden[i] - C_fpga[i]);
        if (diff > 1e-3) { // CHO PHEP SAI SO NHO
            if (errors < 10) { // CHI IN 10 LOI DAU TIEN
                std::cout << "ERROR: C[" << i << "] - Golden: " << C_golden[i]
                          << ", FPGA: " << C_fpga[i] << ", Diff: " << diff << std::endl;
            }
            errors++;
        }
        max_diff = fmaxf(max_diff, diff);
    }

    if (errors == 0) {
        std::cout << "*** TEST PASS ***" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
    } else {
        std::cout << "*** TEST FAIL ***" << std::endl;
        std::cout << "Total errors: " << errors << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
    }


    return (errors == 0) ? 0 : 1;


}
