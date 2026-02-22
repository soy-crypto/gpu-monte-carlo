#include "monte_carlo.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

int main()
{
    // CUDA warmup (create context, eliminate first-call overhead)
    cudaFree(0);

    float S0 = 100.0f;
    float K  = 100.0f;
    float r  = 0.05f;
    float sigma = 0.2f;
    float T = 1.0f;

    int N = 10'000'000;

    auto run_cpu = [&](auto func, const char* name)
    {
        auto start = std::chrono::high_resolution_clock::now();
        float price = func(S0, K, r, sigma, T, N);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(end-start).count();

        std::cout << name << " price=" << price << " time=" << ms << " ms\n";
    };

    run_cpu(monte_carlo_cpu, "CPU");
    run_cpu(monte_carlo_cpu_parallel, "CPU Parallel");

    // GPU: prints kernel-only timing (true compute cost)
    {
        auto start = std::chrono::high_resolution_clock::now();
        float price = monte_carlo_gpu_fast(S0, K, r, sigma, T, N, /*print_kernel_ms=*/true);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(end-start).count();

        std::cout << "GPU Fast (end-to-end) price=" << price << " time=" << ms << " ms\n";
    }

    return 0;
}