// gfx9x MFMA path; gfx11 WMMA path (rocWMMA)
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifdef OSS_USE_WMMA
#include <rocwmma/rocwmma.hpp>
#endif

#ifndef OSS_USE_WMMA
// GEMM: A[M,K] bf16, B[N,K] fp32, C[N,M] fp32 (row-major by N), MFMA path
__global__ void gemm_bf16_f32_mfma_logits(const __hip_bfloat16* __restrict__ A, // [M,K] bf16
                                          const float* __restrict__ B,          // [N,K] f32
                                          float* __restrict__ C,                // [N,M] f32
                                          int M, int N, int K) {
    // Tile geometry
    constexpr int MT_M = 256; // CTA tile height (M)
    constexpr int MT_N = 64;  // CTA tile width  (N)
    constexpr int BK = 64;    // K-slice (multiple of 16)
    constexpr int PAD = 4;
    constexpr int LDS_STRIDE_B = BK + PAD;

    constexpr int WR = 16;                              // waves along M
    constexpr int WC = 1;                               // waves along N
    constexpr int WAVES_PER_CTA = WR * WC;
    constexpr int THREADS_PER_CTA = 64 * WAVES_PER_CTA;

    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int lane_x = lane_id % 16;
    const int lane_y = lane_id / 16;

    const int m0_cta = blockIdx.y * MT_M;
    const int n0_cta = blockIdx.x * MT_N;

    const int m0_wave = (wave_id % WR) * 16;
    const int n0_wave = (wave_id / WR) * 64;

    __shared__ union {
        short sB_T[2][MT_N * LDS_STRIDE_B];
    } smem;

    f4 c0 = {0.f}, c1 = {0.f}, c2 = {0.f}, c3 = {0.f};

    const int num_k_tiles = CEIL_DIV(K, BK);

    // --- Cooperative gmem->smem loader (double-buffered) ---
    auto load_tile_g2s = [&](int k_start, int buf) {
        const int total_vecsB = (BK * MT_N) / 4;
        for (int v = tid; v < total_vecsB; v += THREADS_PER_CTA) {
            int n_loc = v / (BK / 4);
            int k_loc_base = (v % (BK / 4)) * 4;

            int n_gl = n0_cta + n_loc;
            i16x4 bvals = {0, 0, 0, 0};
            if (n_gl < N && k_start + k_loc_base < K) {
                const float* gB = B + (size_t)n_gl * K + k_start + k_loc_base;
                bvals = load_and_convert_f32_to_bf16x4(gB);
            }

            i16x4* sBv =
                reinterpret_cast<i16x4*>(&smem.sB_T[buf][n_loc * LDS_STRIDE_B + k_loc_base]);
            *sBv = bvals;
        }
    };

    load_tile_g2s(0, 0);
    __syncthreads();

    // --- Main loop over K tiles (double buffered) ---
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int ping = kt & 1;
        int pong = ping ^ 1;
        int k0 = kt * BK;

        if (kt + 1 < num_k_tiles) {
            load_tile_g2s(k0 + BK, pong);
        }

        const short* sB_ping = &smem.sB_T[ping][0];
        const int m_gl = m0_cta + m0_wave + lane_x;

        auto load_avec = [&](int kk_offset) {
            i16x4 avec_local = {0, 0, 0, 0};
            if (m_gl < M) {
                int k_gl_base = k0 + kk_offset + lane_y * 4;
                const __hip_bfloat16* gA_ptr = A + (size_t)m_gl * K + k_gl_base;
                if (k_gl_base + 3 < K) {
                    avec_local = *reinterpret_cast<const i16x4*>(gA_ptr);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int k_gl = k_gl_base + i;
                        if (k_gl < K) {
                            avec_local[i] = bf16_bits(gA_ptr[i]);
                        }
                    }
                }
            }
            return avec_local;
        };

        i16x4 avec_curr = load_avec(0);

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            i16x4 avec_next = {0, 0, 0, 0};
            if (kk + 16 < BK)
                avec_next = load_avec(kk + 16);

            const int k_row_base = kk + lane_y * 4;
            const int n_col0 = n0_wave + 0 * 16 + lane_x;
            const int n_col1 = n0_wave + 1 * 16 + lane_x;
            const int n_col2 = n0_wave + 2 * 16 + lane_x;
            const int n_col3 = n0_wave + 3 * 16 + lane_x;

            i16x4 b0 = {0, 0, 0, 0};
            i16x4 b1 = {0, 0, 0, 0};
            i16x4 b2 = {0, 0, 0, 0};
            i16x4 b3 = {0, 0, 0, 0};
            if (n_col0 < MT_N)
                b0 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col0 * LDS_STRIDE_B + k_row_base]);
            if (n_col1 < MT_N)
                b1 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col1 * LDS_STRIDE_B + k_row_base]);
            if (n_col2 < MT_N)
                b2 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col2 * LDS_STRIDE_B + k_row_base]);
            if (n_col3 < MT_N)
                b3 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col3 * LDS_STRIDE_B + k_row_base]);

            c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b0, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b1, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b2, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b3, c3, 0, 0, 0);

            if (kk + 16 < BK)
                avec_curr = avec_next;
        }
        __syncthreads();
    }

    // --- Vectorized writeback to C (N-major: N × M with M contiguous) ---
    const int m_out_base = m0_cta + m0_wave + lane_y * 4;
    const int n_out0 = n0_cta + n0_wave + 0 * 16 + lane_x;
    const int n_out1 = n0_cta + n0_wave + 1 * 16 + lane_x;
    const int n_out2 = n0_cta + n0_wave + 2 * 16 + lane_x;
    const int n_out3 = n0_cta + n0_wave + 3 * 16 + lane_x;

    if (m_out_base + 3 < M) {
        if (n_out0 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out0 * M + m_out_base]) = c0;
        if (n_out1 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out1 * M + m_out_base]) = c1;
        if (n_out2 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out2 * M + m_out_base]) = c2;
        if (n_out3 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out3 * M + m_out_base]) = c3;
    } else {

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int m_out = m_out_base + i;
            if (m_out < M) {
                if (n_out0 < N)
                    nt_store(&C[(size_t)n_out0 * M + m_out], c0[i]);
                if (n_out1 < N)
                    nt_store(&C[(size_t)n_out1 * M + m_out], c1[i]);
                if (n_out2 < N)
                    nt_store(&C[(size_t)n_out2 * M + m_out], c2[i]);
                if (n_out3 < N)
                    nt_store(&C[(size_t)n_out3 * M + m_out], c3[i]);
            }
        }
    }
}
#else

// Simple WMMA-based GEMM kernel for gfx11: C[N,M] = A[M,K] (bf16) @ B[N,K]^T (f32 -> bf16)
// Operates on 16x16x16 tiles, one wave32 per tile. Stages B as bf16 in LDS.
__device__ __forceinline__ hip_bfloat16 f32_to_bf16(float val) { return hip_bfloat16(val); }

__global__ void gemm_bf16_f32_wmma_logits(const __hip_bfloat16* __restrict__ A, // [M,K] bf16
                                          const float* __restrict__ B,          // [N,K] f32
                                          float* __restrict__ C,                // [N,M] f32
                                          int M, int N, int K) {
    using namespace rocwmma;
    constexpr int TM = 16, TN = 16, TK = 16;

    // tile indices: grid.x covers M in tiles; grid.y covers N in tiles
    int tile_m = blockIdx.x * TM;
    int tile_n = blockIdx.y * TN;

    // Accumulator for 16x16 tile (float32)
    fragment<accumulator, TM, TN, TK, float32_t> acc;
    fill_fragment(acc, static_cast<float32_t>(0.0f));

    extern __shared__ char smem[];
    hip_bfloat16* sA = reinterpret_cast<hip_bfloat16*>(smem);                        // [TN, TK]
    hip_bfloat16* sB = sA + (TN * TK);                                              // [TK, TM]

    // Loop over K in 16-chunks
    for (int k0 = 0; k0 < K; k0 += TK) {
        // Stage A_tile: from B[N,K] -> sA[Ntile, Ktile] as bf16
        // Stage B_tile: from A[M,K] -> sB[Ktile, Mtile] as bf16
        for (int idx = threadIdx.x; idx < TN * TK; idx += blockDim.x) {
            int r = idx / TK; // 0..15 (Ntile)
            int c = idx % TK; // 0..15 (Ktile)
            hip_bfloat16 v = hip_bfloat16(0.0f);
            int n = tile_n + r;
            int k = k0 + c;
            if (n < N && k < K) {
                v = f32_to_bf16(B[(size_t)n * K + k]);
            }
            sA[r * TK + c] = v;
        }
        for (int idx = threadIdx.x; idx < TK * TM; idx += blockDim.x) {
            int r = idx / TM; // 0..15 (Ktile)
            int c = idx % TM; // 0..15 (Mtile)
            hip_bfloat16 v = hip_bfloat16(0.0f);
            int m = tile_m + c;
            int k = k0 + r;
            if (m < M && k < K) {
                v = hip_bfloat16(__bfloat162float(A[(size_t)m * K + k]));
            }
            sB[r * TM + c] = v;
        }
        __syncthreads();

        // Load fragments from LDS
        fragment<matrix_a, TM, TN, TK, bfloat16_t, row_major> a_frag;
        fragment<matrix_b, TM, TN, TK, bfloat16_t, row_major> b_frag;
        load_matrix_sync(a_frag, sA, TK);
        load_matrix_sync(b_frag, sB, TM);

        // MMA
        mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    // Store accumulator to global C (row-major N x M)
    // We use a temp fragment for store
    fragment<accumulator, TM, TN, TK, float32_t> c_frag = acc;
    // Bounds-safe store: use a guarded path
    for (int i = 0; i < c_frag.num_elements; ++i) {
        // Use rocWMMA store helper into a temp tile in shared, then scatter
    }

    // Use rocWMMA store_matrix_sync to a temporary scratch (reuse s_mem)
    float* sC = reinterpret_cast<float*>(sB + (TK * TM));
    __syncthreads();
    store_matrix_sync(sC, c_frag, TM, mem_row_major); // sC is [TN x TM] row-major by TM
    __syncthreads();

    // Scatter to global with bounds checks
    for (int idx = threadIdx.x; idx < TN * TM; idx += blockDim.x) {
        int r = idx / TM; // 0..15 (Ntile)
        int c = idx % TM; // 0..15 (Mtile)
        int n = tile_n + r;
        int m = tile_m + c;
        if (n < N && m < M) {
            C[(size_t)n * M + m] = sC[r * TM + c];
        }
    }
}

void gemm_logits(float* xout, const float* x_batch, const __hip_bfloat16* w, int N, int K, int M, hipStream_t stream) {
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dim3 block(32, 1, 1);
    size_t shmem = (16 * 16 + 16 * 16) * sizeof(hip_bfloat16) + 16 * 16 * sizeof(float);
    hipLaunchKernelGGL(gemm_bf16_f32_wmma_logits, grid, block, shmem, stream, w, x_batch, xout, M, N, K);
}
#endif

#ifndef OSS_USE_WMMA
void gemm_logits(float* xout, const float* x_batch, const __hip_bfloat16* w, int N, int K, int M, hipStream_t stream) {
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
    dim3 block(1024, 1, 1);
    gemm_bf16_f32_mfma_logits<<<grid, block, 0, stream>>>(w, x_batch, xout, M, N, K);
}
#endif
