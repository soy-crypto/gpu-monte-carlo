#include "monte_carlo.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__
void mc_kernel_optimized(
    float *block_sums,
    float S0, float K,
    float r, float sigma,
    float T, int N)
{
    __shared__ float shared[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float payoff = 0.0f;

    if(idx < N)
    {
        curandState state;
        curand_init(42, idx, 0, &state);

        float Z = curand_normal(&state);

        float ST = S0 * __expf(
            (r - 0.5f*sigma*sigma)*T +
            sigma * sqrtf(T) * Z
        );

        payoff = fmaxf(ST - K, 0.0f);
    }

    shared[tid] = payoff;
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if(tid == 0)
        block_sums[blockIdx.x] = shared[0];
}

float monte_carlo_gpu_optimized(
    float S0, float K, float r,
    float sigma, float T, int N)
{
    int block = BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    float *d_block_sums;
    cudaMalloc(&d_block_sums,
               grid*sizeof(float));

    mc_kernel_optimized<<<grid, block>>>(
        d_block_sums,
        S0, K, r, sigma, T, N);

    float *h_block_sums =
        new float[grid];

    cudaMemcpy(h_block_sums,
               d_block_sums,
               grid*sizeof(float),
               cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for(int i=0;i<grid;i++)
        sum += h_block_sums[i];

    cudaFree(d_block_sums);
    delete[] h_block_sums;

    return exp(-r*T) * sum / N;
}