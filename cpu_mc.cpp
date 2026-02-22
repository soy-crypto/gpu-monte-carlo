#include "monte_carlo.h"
#include <random>
#include <cmath>
#include <omp.h>

float monte_carlo_cpu(
    float S0, float K, float r,
    float sigma, float T, int N)
{
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float payoff_sum = 0.0f;

    for(int i = 0; i < N; i++)
    {
        float Z = dist(gen);

        float ST = S0 * exp(
            (r - 0.5f*sigma*sigma)*T +
            sigma * sqrt(T) * Z
        );

        payoff_sum += fmax(ST - K, 0.0f);
    }

    return exp(-r*T) * payoff_sum / N;
}

float monte_carlo_cpu_parallel(
    float S0, float K, float r,
    float sigma, float T, int N)
{
    float payoff_sum = 0.0f;

    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        float local_sum = 0.0f;

        #pragma omp for
        for(int i = 0; i < N; i++)
        {
            float Z = dist(gen);

            float ST = S0 * exp(
                (r - 0.5f*sigma*sigma)*T +
                sigma * sqrt(T) * Z
            );

            local_sum += fmax(ST - K, 0.0f);
        }

        #pragma omp atomic
        payoff_sum += local_sum;
    }

    return exp(-r*T) * payoff_sum / N;
}