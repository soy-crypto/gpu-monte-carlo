#include "monte_carlo.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>

__global__
void mc_kernel(
    float *results,
    float S0, float K,
    float r, float sigma,
    float T, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N) return;

    curandState state;
    curand_init(42, idx, 0, &state);

    float Z = curand_normal(&state);

    float ST = S0 * __expf(
        (r - 0.5f*sigma*sigma)*T +
        sigma * sqrtf(T) * Z
    );

    results[idx] = fmaxf(ST - K, 0.0f);
}

float monte_carlo_gpu_naive(
    float S0, float K, float r,
    float sigma, float T, int N)
{
    float *d_results;
    cudaMalloc(&d_results, N * sizeof(float));

    int block = 256;
    int grid = (N + block - 1) / block;

    mc_kernel<<<grid, block>>>(
        d_results, S0, K, r, sigma, T, N);

    float *h_results = new float[N];
    cudaMemcpy(h_results, d_results,
               N*sizeof(float),
               cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for(int i=0;i<N;i++)
        sum += h_results[i];

    cudaFree(d_results);
    delete[] h_results;

    return exp(-r*T) * sum / N;
}