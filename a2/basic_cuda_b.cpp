#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>

__device__ double *d_y;                    //*d_y now is a uninitialized variable on GPU memory
__global__ void initMemory(int vectorsize) 
{
    cudaMalloc(&d_y, sizeof(double) * vectorsize); // allocate memory
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = id; i < vectorsize; i += blockDim.x * gridDim.x) // initiate
    {
        d_y[i] = i;
    }
}

    main(int argc, char **argv)
{
    // initicpyize vector lengths
    std::vector<int> n = {100, 1000, 10000, 100000, 1000000, 10000000};
    // save data struc
    std::vector<std::vector<float>> times_N;
    Timer timer;
    // outer loop over vector lengths
    for (int j = 0; j < n.size(); j++)
    {
        int N = n[j];

        // save data struc
        std::vector<float> log_cpy;
        std::vector<float> log_fr;

        for (int i = 0; i < 10; i++)
        {
            // allocate device memory and initiate via copy
            double *x, *d_x;
            cudaDeviceSynchronize();
            timer.reset();
            x = (double *)malloc(N * sizeof(double));
            cudaMalloc(&d_x, N * sizeof(double));
            for (int k = 0; k < N; k++)
            {
                x[k] = 1.0;
            }
            cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            float elapsed_time_cpy = timer.get();
            log_cpy.push_back(elapsed_time_cpy);

            // allociate device memory and initiate in kernel
            cudaDeviceSynchronize();
            timer.reset();
            initMemory<<<256,256>>>(N);
            cudaDeviceSynchronize();
            float elapsed_time_fr = timer.get();
            log_fr.push_back(elapsed_time_fr);

            free(x);
            cudaFree(d_x);
            double *del;
            cudaMemcpyFromSymbol(&del, d_y, sizeof(del), 0, cudaMemcpyDeviceToHost);
            cudaFree(del);
        }

        // build averages
        float log_cpy_av = std::accumulate(log_cpy.begin(), log_cpy.end(), 0.0 / log_cpy.size());
        float log_fr_av = std::accumulate(log_fr.begin(), log_fr.end(), 0.0 / log_fr.size());

        // N t_cpy t_fr
        std::cout << N << " " << log_cpy_av << " " << log_fr_av << std::endl;

        std::vector<float> time_N;
        time_N.push_back(log_cpy_av);
        time_N.push_back(log_fr_av);

        times_N.push_back(time_N);
    }

    return EXIT_SUCCESS;
}