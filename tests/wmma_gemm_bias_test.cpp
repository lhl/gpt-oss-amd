#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>

// Prototypes
void gemm_qkv(float* xout, const float* x_batch, const __hip_bfloat16* w,
              const __hip_bfloat16* bias, int N, int K, int M, hipStream_t stream);
void gemm_o(float* xout, const float* x_batch, const __hip_bfloat16* w, const __hip_bfloat16* bias,
            int N, int K, int M, hipStream_t stream);
void gemm_router(float* xout, const float* x_batch, const __hip_bfloat16* w,
                 const __hip_bfloat16* bias, int N, int K, int M, hipStream_t stream);

static inline __hip_bfloat16 f2b(float v) { return __float2bfloat16(v); }
static inline float b2f(__hip_bfloat16 v) { return __bfloat162float(v); }

struct Case { int N, K, M; };

template <typename GemmFn>
int run_one(const char* name, GemmFn fn, int N, int K, int M) {
    std::vector<float> B(N * K);
    std::vector<__hip_bfloat16> W(M * K);
    std::vector<__hip_bfloat16> bias(M);
    std::vector<float> ref(N * M, 0.0f);
    std::vector<float> out(N * M, 0.0f);

    srand(123 + N + K + M);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            B[n * K + k] = ((rand() % 1000) - 500) / 500.0f;
    for (int m = 0; m < M; ++m) {
        bias[m] = f2b(((rand() % 1000) - 500) / 1000.0f);
        for (int k = 0; k < K; ++k)
            W[m * K + k] = f2b(((rand() % 1000) - 500) / 500.0f);
    }

    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)B[n * K + k] * (double)b2f(W[m * K + k]);
            ref[n * M + m] = (float)acc + b2f(bias[m]);
        }
    }

    float *dB = nullptr, *dC = nullptr; __hip_bfloat16 *dW = nullptr, *dBias = nullptr;
    hipMalloc(&dB, N * K * sizeof(float));
    hipMalloc(&dC, N * M * sizeof(float));
    hipMalloc(&dW, M * K * sizeof(__hip_bfloat16));
    hipMalloc(&dBias, M * sizeof(__hip_bfloat16));
    hipMemcpy(dB, B.data(), N * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dW, W.data(), M * K * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(dBias, bias.data(), M * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);

    fn(dC, dB, dW, dBias, N, K, M, 0);
    hipDeviceSynchronize();
    hipMemcpy(out.data(), dC, N * M * sizeof(float), hipMemcpyDeviceToHost);

    // Compare
    int mismatches = 0; double max_abs = 0.0;
    for (int i = 0; i < N * M; ++i) {
        double e = std::fabs(out[i] - ref[i]);
        if (e > 2e-1) {
            mismatches++; if (mismatches < 5) fprintf(stderr, "%s mismatch @%d got %.6f ref %.6f\n", name, i, out[i], ref[i]);
        }
        if (e > max_abs) max_abs = e;
    }
    printf("%s: mismatches=%d, max_abs_err=%.6f (N=%d K=%d M=%d)\n", name, mismatches, max_abs, N, K, M);

    hipFree(dB); hipFree(dC); hipFree(dW); hipFree(dBias);
    return mismatches;
}

int main() {
    std::vector<Case> cases = {{32, 48, 64}, {33, 48, 63}, {17, 80, 19}};
    int total = 0;
    for (auto c : cases) total += run_one("gemm_qkv", gemm_qkv, c.N, c.K, c.M);
    for (auto c : cases) total += run_one("gemm_o", gemm_o, c.N, c.K, c.M);
    for (auto c : cases) total += run_one("gemm_router", gemm_router, c.N, c.K, c.M);
    return total == 0 ? 0 : 1;
}
