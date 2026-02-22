#include "monte_carlo.h"
#include <iostream>
#include <chrono>

int main()
{
    float S0 = 100.0f;
    float K  = 100.0f;
    float r  = 0.05f;
    float sigma = 0.2f;
    float T = 1.0f;

    int N = 10'000'000;

    auto run = [&](auto func,
                   const char* name)
    {
        auto start =
        std::chrono::high_resolution_clock::now();

        float price =
        func(S0,K,r,sigma,T,N);

        auto end =
        std::chrono::high_resolution_clock::now();

        double ms =
        std::chrono::duration<double,std::milli>
        (end-start).count();

        std::cout << name
                  << " price=" << price
                  << " time=" << ms
                  << " ms\n";
    };

    run(monte_carlo_cpu, "CPU");
    run(monte_carlo_cpu_parallel, "CPU Parallel");
    run(monte_carlo_gpu_naive, "GPU Naive");
    run(monte_carlo_gpu_optimized, "GPU Optimized");

    return 0;
}