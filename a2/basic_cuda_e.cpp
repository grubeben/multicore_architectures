#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>

__global__ void add(int n, double *x, double *y, double *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = x[i] + y[i];
}

int main(int argc, char **argv)
{
    // initicpyize vector lengths
    std::vector<int> thread_options = {16, 32, 64, 128, 256, 512, 1024};

    Timer timer;

    int N = 10000000;

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

    // outer loop over vector lengths
    for (int j = 0; j < thread_options.size(); j++)
    {
        for (int m = 0; m < thread_options.size(); m++)
        {
            // save data struc
            std::vector<float> log_add;

            int n_threads = thread_options[j];
            int n_blocks = thread_options[m];

            for (int i = 0; i < 10; i++)
            {
                // + operation
                cudaDeviceSynchronize();
                timer.reset();
                //<<<<n_blocks, n_threads/block>>>
                add<<<n_blocks, n_threads>>>(N, d_x, d_y, d_z);
                cudaDeviceSynchronize();
                float elapsed_time_add = timer.get();
                log_add.push_back(elapsed_time_add);
            }
            // copy data back (implicit synchronization point)
            cudaMemcpy(z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost);

            // build averages
            float log_add_av = std::accumulate(log_add.begin(), log_add.end(), 0.0 / log_add.size());

            // output
            std::cout << n_blocks << " " << n_threads << " " << log_add_av << std::endl;
        }
    }
        
    free(x);
    free(y);
    free(z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return EXIT_SUCCESS;
}