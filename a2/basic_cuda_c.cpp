#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>

__global__ void add(int n, double *x, double *y, double *z)
{
    const int total_threads = blockDim.x * gridDim.x; // number of threads
    int id = blockIdx.x * blockDim.x + threadIdx.x;   // #block    #threads/block   #offset within block

    for (unsigned int i = id; i < n; i += total_threads)
    {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char **argv)
{
    // initicpyize vector lengths
    std::vector<int> n = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000};
    // save data struc
    std::vector<std::vector<float>> times_N;
    Timer timer;

    // outer loop over vector lengths
    for (int j = 0; j < n.size(); j++)
    {
        int N = n[j];

        // allocate device memory and initiate via copy
        double *x, *d_x, *y, *d_y, *z, *d_z;
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        z = (double *)malloc(N * sizeof(double));
        cudaMalloc(&d_x, N * sizeof(double));
        cudaMalloc(&d_y, N * sizeof(double));
        cudaMalloc(&d_z, N * sizeof(double));
        for (int k = 0; k < N; k++)
        {
            x[k] = 1.0;
            y[k] = 2.0;
            z[k] = 0.0;
        }
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, z, N * sizeof(double), cudaMemcpyHostToDevice);

        // save data struc
        std::vector<float> log_cpy;
        std::vector<float> log_add;

        for (int i = 0; i < 10; i++)
        {
            // + operation
            cudaDeviceSynchronize();
            timer.reset();
            int n_threads = 256;
            int n_blocks = (N + n_threads - 1) / n_threads;
            add<<<n_blocks, n_threads>>>(N, d_x, d_y, d_z);
            cudaDeviceSynchronize();
            float elapsed_time_add = timer.get();
            log_add.push_back(elapsed_time_add);
        }
        // copy data back (implicit synchronization point)
        cudaMemcpy(z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost);

        // build averages
        //float log_cpy_av = std::accumulate(log_cpy.begin(), log_cpy.end(), 0.0 / log_cpy.size());
        float log_add_av = std::accumulate(log_add.begin(), log_add.end(), 0.0 / log_add.size());

        // N t_cpy t_fr
        std::cout << N << " " << log_add_av << std::endl;

        std::vector<float> time_N;
        //time_N.push_back(log_cpy_av);
        time_N.push_back(log_add_av);

        times_N.push_back(time_N);

        free(x);
        free(y);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    return EXIT_SUCCESS;
}