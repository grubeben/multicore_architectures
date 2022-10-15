

#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char **argv)
{
    // initialize vector lengths
    std::vector<int> n = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000};

    // outer loop over vector lengths
    for (int j = 0; j < n.size(); j++)
    {
        int N = n[j];
        double *d_x;
        Timer timer;

        // save data struc
        std::vector<float> log_al;
        std::vector<float> log_fr;

        for (int i = 0; i < 10; i++)
        {
            // Allocate device memory
            cudaDeviceSynchronize();
            timer.reset();
            cudaMalloc(&d_x, N * sizeof(double));
            cudaDeviceSynchronize();
            float elapsed_time_al = timer.get();
            log_al.push_back(elapsed_time_al);

            // free memory
            cudaDeviceSynchronize();
            timer.reset();
            cudaFree(d_x);
            cudaDeviceSynchronize();
            float elapsed_time_fr = timer.get();
            log_fr.push_back(elapsed_time_fr);
        }

        // build averages
        float log_al_av = std::accumulate(log_al.begin(), log_al.end(), 0.0 / log_al.size());
        float log_fr_av = std::accumulate(log_fr.begin(), log_fr.end(), 0.0 / log_fr.size());

        // N t_al t_fr
        std::cout << N << " " << log_al_av << " " << log_fr_av << std::endl;
    }

    return EXIT_SUCCESS;
}
