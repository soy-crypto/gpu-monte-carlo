#include "monte_carlo.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>

static inline void checkCuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s: %s\n",
                msg, cudaGetErrorString(e));
        throw std::runtime_error("CUDA failure");
    }
}

// ------------------------------------------------------------
// 4 PATHS PER THREAD + BLOCK REDUCTION
// ------------------------------------------------------------

#define BLOCK_SIZE 256

__global__ void init_rng_states(
    curandStatePhilox4_32_10_t* states,
    int N4,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N4) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void mc_payoff_reduce_kernel4(
    float* block_sums,
    curandStatePhilox4_32_10_t* states,
    float S0, float K,
    float r, float sigma,
    float T, int N)
{
    __shared__ float shared[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;
    int tid = threadIdx.x;

    float sumPayoff = 0.0f;

    if (base < N)
    {
        curandStatePhilox4_32_10_t local = states[idx];
        float4 z4 = curand_normal4(&local);
        states[idx] = local;

        float drift = (r - 0.5f * sigma * sigma) * T;
        float sigT  = sigma * sqrtf(T);

        float Zs[4] = {z4.x, z4.y, z4.z, z4.w};

        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int path = base + j;
            if (path < N)
            {
                float ST = S0 * __expf(drift + sigT * Zs[j]);
                sumPayoff += fmaxf(ST - K, 0.0f);
            }
        }
    }

    shared[tid] = sumPayoff;
    __syncthreads();

    // Block reduction (shared memory)
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        block_sums[blockIdx.x] = shared[0];
}

// ------------------------------------------------------------
// Persistent GPU Engine
// ------------------------------------------------------------

struct GpuMcEngine
{
    int N = 0;
    int N4 = 0;
    int block = BLOCK_SIZE;
    int grid = 0;

    float* d_block_sums = nullptr;
    curandStatePhilox4_32_10_t* d_states = nullptr;

    bool initialized = false;

    void init(int newN)
    {
        if (initialized && newN == N) return;

        release();

        N = newN;
        N4 = (N + 3) / 4;
        grid = (N4 + block - 1) / block;

        checkCuda(cudaMalloc(&d_block_sums,
                             grid * sizeof(float)),
                  "cudaMalloc d_block_sums");

        checkCuda(cudaMalloc(&d_states,
                             N4 * sizeof(curandStatePhilox4_32_10_t)),
                  "cudaMalloc d_states");

        init_rng_states<<<grid, block>>>(
            d_states, N4, 42ULL);

        checkCuda(cudaGetLastError(),
                  "launch init_rng_states");

        checkCuda(cudaDeviceSynchronize(),
                  "sync init_rng_states");

        initialized = true;
    }

    void release()
    {
        if (d_states) cudaFree(d_states);
        if (d_block_sums) cudaFree(d_block_sums);

        d_states = nullptr;
        d_block_sums = nullptr;
        initialized = false;
        N = 0;
        N4 = 0;
        grid = 0;
    }

    ~GpuMcEngine()
    {
        release();
    }
};

static GpuMcEngine& engine()
{
    static GpuMcEngine e;
    return e;
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

float monte_carlo_gpu_fast(
    float S0, float K,
    float r, float sigma,
    float T, int N,
    bool print_kernel_ms)
{
    auto& e = engine();
    e.init(N);

    cudaEvent_t start, stop;
    float kernel_ms = 0.0f;

    if (print_kernel_ms)
    {
        checkCuda(cudaEventCreate(&start),
                  "event create start");
        checkCuda(cudaEventCreate(&stop),
                  "event create stop");
        checkCuda(cudaEventRecord(start),
                  "event record start");
    }

    mc_payoff_reduce_kernel4<<<e.grid, e.block>>>(
        e.d_block_sums,
        e.d_states,
        S0, K, r, sigma, T, N);

    checkCuda(cudaGetLastError(),
              "launch mc_payoff_reduce_kernel4");

    if (print_kernel_ms)
    {
        checkCuda(cudaEventRecord(stop),
                  "event record stop");
        checkCuda(cudaEventSynchronize(stop),
                  "event sync stop");
        checkCuda(cudaEventElapsedTime(
                  &kernel_ms,
                  start, stop),
                  "event elapsed");

        printf("GPU kernel-only time: %.3f ms\n",
               kernel_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy only block sums
    std::vector<float> h_block(e.grid);

    checkCuda(cudaMemcpy(
        h_block.data(),
        e.d_block_sums,
        e.grid * sizeof(float),
        cudaMemcpyDeviceToHost),
        "memcpy block sums");

    float total = 0.0f;
    for (int i = 0; i < e.grid; i++)
        total += h_block[i];

    return expf(-r * T) *
           (total / static_cast<float>(N));
}