#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

// Prototype from our GEMM implementation
void gemm_logits(float* xout, const float* x_batch, const __hip_bfloat16* w, int N, int K, int M, hipStream_t stream);

static inline __hip_bfloat16 f32_to_bf16(float v) { return __float2bfloat16(v); }
static inline float bf16_to_f32(__hip_bfloat16 v) { return __bfloat162float(v); }

int main() {
    const int M = 32;  // vocab or out features
    const int N = 33;  // batch
    const int K = 48;  // hidden

    std::vector<float> hB(N * K);
    std::vector<__hip_bfloat16> hA(M * K);
    std::vector<float> href(N * M, 0.0f);
    std::vector<float> hout(N * M, 0.0f);

    // Seeded values
    srand(42);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = ((rand() % 1000) - 500) / 500.0f; // [-1, 1]
            hB[n * K + k] = v;
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float v = ((rand() % 1000) - 500) / 500.0f; // [-1, 1]
            hA[m * K + k] = f32_to_bf16(v);
        }
    }

    // CPU reference: C[n,m] = sum_k B[n,k] * A[m,k]
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)hB[n * K + k] * (double)bf16_to_f32(hA[m * K + k]);
            }
            href[n * M + m] = (float)acc;
        }
    }

    // Device alloc
    float *dB = nullptr, *dC = nullptr;
    __hip_bfloat16* dA = nullptr;
    hipMalloc(&dB, N * K * sizeof(float));
    hipMalloc(&dC, N * M * sizeof(float));
    hipMalloc(&dA, M * K * sizeof(__hip_bfloat16));
    hipMemcpy(dB, hB.data(), N * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dA, hA.data(), M * K * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);

    gemm_logits(dC, dB, dA, N, K, M, 0);
    hipDeviceSynchronize();

    hipMemcpy(hout.data(), dC, N * M * sizeof(float), hipMemcpyDeviceToHost);

    // Compare
    int mismatches = 0;
    double max_abs_err = 0.0;
    for (int i = 0; i < N * M; ++i) {
        double e = std::fabs((double)hout[i] - (double)href[i]);
        if (e > 2e-1) {
            mismatches++;
            if (mismatches < 10) {
                fprintf(stderr, "mismatch at %d: got %.6f, ref %.6f (abs err %.6f)\n", i, hout[i], href[i], e);
            }
        }
        if (e > max_abs_err) max_abs_err = e;
    }
    printf("WMMA GEMM test: mismatches=%d, max_abs_err=%.6f\n", mismatches, max_abs_err);

    hipFree(dB);
    hipFree(dC);
    hipFree(dA);

    return mismatches == 0 ? 0 : 1;
}

