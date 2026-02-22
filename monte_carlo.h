#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

float monte_carlo_cpu(float S0, float K, float r, float sigma, float T, int N);
float monte_carlo_cpu_parallel(float S0, float K, float r, float sigma, float T, int N);

// Fast GPU version: persistent RNG + device reduction (CUB)
float monte_carlo_gpu_fast(float S0, float K, float r, float sigma, float T, int N, bool print_kernel_ms);

#endif