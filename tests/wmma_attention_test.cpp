// Minimal attention correctness test against CPU reference
#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>

#include <random>
#include <hip/hip_bfloat16.h>
#include <vector>

// Attention kernel entry
extern "C" void fa_run(const float* q_batch, const void* k_cache, const void* v_cache,
                       const float* mask, const void* attn_sinks, float* tb_batch, int B,
                       int seq_len, int head_dim, int kv_dim, int kv_mul, int sliding_window,
                       int layer_idx, int n_attn_heads, int kv_cache_is_fp16,
                       const int* d_pos_per_token, const int* d_batch_indices, long long B_stride,
                       int max_pos_in_batch, hipStream_t stream, const long long* d_layer_kv_off,
                       const int* d_layer_kv_cap, const int* d_layer_is_local);

static void checkHip(hipError_t e, const char* msg) {
    if (e != hipSuccess) {
        fprintf(stderr, "HIP error %s: %s\n", msg, hipGetErrorString(e));
        std::abort();
    }
}

// CPU reference for causal attention without mask/sinks
static void cpu_attn(float* out, const float* q_batch, const float* k_cache, const float* v_cache,
                     int B, int T, int head_dim, int n_attn_heads, int n_kv_heads, int kv_mul) {
    const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    const int kv_dim = head_dim * n_kv_heads;
    for (int b = 0; b < B; ++b) {
        int pos = T - 1; // use full prefix
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            for (int r = 0; r < kv_mul; ++r) {
                int h = kvh * kv_mul + r;
                if (h >= n_attn_heads) continue;
                const float* q = q_batch + (size_t)b * n_attn_heads * head_dim + h * head_dim;
                std::vector<float> scores(pos + 1);
                for (int t = 0; t <= pos; ++t) {
                    const float* k = k_cache + (size_t)b * T * kv_dim + t * kv_dim + kvh * head_dim;
                    float s = 0.f;
                    for (int d = 0; d < head_dim; ++d) s += q[d] * inv_sqrt_d * k[d];
                    scores[t] = s;
                }
                // softmax
                float m = -INFINITY;
                for (int t = 0; t <= pos; ++t) m = std::max(m, scores[t]);
                float Z = 0.f;
                for (int t = 0; t <= pos; ++t) { scores[t] = std::exp(scores[t] - m); Z += scores[t]; }
                for (int t = 0; t <= pos; ++t) scores[t] /= (Z > 0 ? Z : 1);
                // aggregate V
                float* o = out + (size_t)b * n_attn_heads * head_dim + h * head_dim;
                for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
                for (int t = 0; t <= pos; ++t) {
                    const float* v = v_cache + (size_t)b * T * kv_dim + t * kv_dim + kvh * head_dim;
                    float w = scores[t];
                    for (int d = 0; d < head_dim; ++d) o[d] += w * v[d];
                }
            }
        }
    }
}

int main() {
    // Shapes
    const int B = 2;              // batch
    const int T = 32;             // seq_len
    const int head_dim = 64;
    const int n_attn_heads = 2;   // two attention heads
    const int n_kv_heads = 1;     // one KV head
    const int kv_mul = 2;         // 2 attn per KV
    const int kv_dim = head_dim * n_kv_heads;

    // Host init
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> hQ((size_t)B * n_attn_heads * head_dim);
    std::vector<float> hK((size_t)B * T * kv_dim);
    std::vector<float> hV((size_t)B * T * kv_dim);
    for (auto& x : hQ) x = dist(rng);
    for (auto& x : hK) x = dist(rng);
    for (auto& x : hV) x = dist(rng);
    std::vector<float> hOut((size_t)B * n_attn_heads * head_dim, 0.0f);
    std::vector<float> hRef((size_t)B * n_attn_heads * head_dim, 0.0f);

    // Device buffers
    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dOut=nullptr;
    int *dPos=nullptr, *dBatchIdx=nullptr;
    long long *dLayerOff=nullptr; int *dLayerCap=nullptr, *dIsLocal=nullptr;
    checkHip(hipMalloc(&dQ, hQ.size()*sizeof(float)), "malloc dQ");
    checkHip(hipMalloc(&dK, hK.size()*sizeof(float)), "malloc dK");
    checkHip(hipMalloc(&dV, hV.size()*sizeof(float)), "malloc dV");
    checkHip(hipMalloc(&dOut, hOut.size()*sizeof(float)), "malloc dOut");
    std::vector<int> hPos(B, T-1), hBatch(B); for (int i=0;i<B;++i) hBatch[i]=i;
    long long hLayerOff[1] = {0}; int hLayerCap[1] = {T}; int hIsLocal[1] = {0};
    checkHip(hipMalloc(&dPos, B*sizeof(int)), "malloc dPos");
    checkHip(hipMalloc(&dBatchIdx, B*sizeof(int)), "malloc dBatchIdx");
    checkHip(hipMalloc(&dLayerOff, sizeof(long long)), "malloc dLayerOff");
    checkHip(hipMalloc(&dLayerCap, sizeof(int)), "malloc dLayerCap");
    checkHip(hipMalloc(&dIsLocal, sizeof(int)), "malloc dIsLocal");
    checkHip(hipMemcpy(dQ, hQ.data(), hQ.size()*sizeof(float), hipMemcpyHostToDevice), "cpy Q");
    checkHip(hipMemcpy(dK, hK.data(), hK.size()*sizeof(float), hipMemcpyHostToDevice), "cpy K");
    checkHip(hipMemcpy(dV, hV.data(), hV.size()*sizeof(float), hipMemcpyHostToDevice), "cpy V");
    checkHip(hipMemcpy(dPos, hPos.data(), B*sizeof(int), hipMemcpyHostToDevice), "cpy Pos");
    checkHip(hipMemcpy(dBatchIdx, hBatch.data(), B*sizeof(int), hipMemcpyHostToDevice), "cpy BatchIdx");
    checkHip(hipMemcpy(dLayerOff, hLayerOff, sizeof(long long), hipMemcpyHostToDevice), "cpy LayerOff");
    checkHip(hipMemcpy(dLayerCap, hLayerCap, sizeof(int), hipMemcpyHostToDevice), "cpy LayerCap");
    checkHip(hipMemcpy(dIsLocal, hIsLocal, sizeof(int), hipMemcpyHostToDevice), "cpy IsLocal");

    hipStream_t stream=nullptr;
    fa_run(dQ, dK, dV, /*mask*/nullptr, /*sinks*/nullptr, dOut,
       B, T, head_dim, kv_dim, kv_mul, /*sliding_window*/0, /*layer_idx*/0, n_attn_heads,
       /*kv_cache_is_fp16*/0, dPos, dBatchIdx, /*B_stride*/0, /*max_pos_in_batch*/T-1, stream,
       dLayerOff, dLayerCap, dIsLocal);
    checkHip(hipDeviceSynchronize(), "sync");
    checkHip(hipMemcpy(hOut.data(), dOut, hOut.size()*sizeof(float), hipMemcpyDeviceToHost), "cpy Out");

    // CPU reference
    cpu_attn(hRef.data(), hQ.data(), hK.data(), hV.data(), B, T, head_dim, n_attn_heads, n_kv_heads, kv_mul);

    // Compare
    int mismatches=0; float max_abs_err=0.0f;
    for (size_t i=0;i<hOut.size();++i) {
        float e = std::fabs(hOut[i] - hRef[i]);
        if (e > 3e-2f) ++mismatches;
        if (e > max_abs_err) max_abs_err = e;
    }
    std::printf("wmma_attention_test: mismatches=%d, max_abs_err=%g\n", mismatches, max_abs_err);

    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dOut);
    hipFree(dPos); hipFree(dBatchIdx); hipFree(dLayerOff); hipFree(dLayerCap); hipFree(dIsLocal);
    return 0;
}
