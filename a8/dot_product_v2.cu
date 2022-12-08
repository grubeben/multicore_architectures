#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// step 1: partial product sums
__global__ void dot_s1(int n, double *x, double *y, double *z_scalar)
{
    __shared__ double temp[256];                      // shared temporal memory on GPU RAM vs. extern __shared__: this is used in case size of temp is unclear
    double thread_pprod = 0;                          // local for thread
    const int total_threads = blockDim.x * gridDim.x; // number of threads
    int id = blockIdx.x * blockDim.x + threadIdx.x;   // #block    #threads/block   #offset within block

    for (unsigned int i = id; i < n; i += total_threads)
    {
        thread_pprod += x[i] * y[i];
    }

    // parallel primitives - stride
    temp[threadIdx.x] = thread_pprod;
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            temp[threadIdx.x] += temp[threadIdx.x + stride];
    }

    // only one thread/block writes results into z_scalar vector
    if (threadIdx.x == 0)
    {
        z_scalar[blockIdx.x] = temp[0];
    }
}

// step 2: intitialise only one block to sums up z_scalar entries
__global__ void dot_s2(double *z_scalar)
{

    // parallel primitives - stride
    // temp[threadIdx.x] = z_scalar[threadIdx.x]
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            z_scalar[threadIdx.x] += z_scalar[threadIdx.x + stride];
        }
    }
    // since we only have one block temp[0] holds our sum now
}

int main(void)
{
    Timer timer;                                                                                 // load timer

    for (int j = 32; j < 1e8; j*=2) // outer loop over vector lengths
    {
        int N = j;

        // allocate device memory and initiate via copy
        double *x, *d_x, *y, *d_y, *z_vec, *d_z_vec, *z_scalar, *d_z_scalar;
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        z_vec = (double *)malloc(256 * sizeof(double));
        z_scalar = (double *)malloc(256 * sizeof(double));

        cudaMalloc(&d_x, N * sizeof(double));
        cudaMalloc(&d_y, N * sizeof(double));
        cudaMalloc(&d_z_vec, 256 * sizeof(double));
        cudaMalloc(&d_z_scalar, 256 * sizeof(double));

        for (int k = 0; k < N; k++)
        {
            x[k] = 1.0;
            y[k] = 1.0;
        }

        for (int k = 0; k < 256; k++)
        {
            z_vec[k] = 0.0;
            z_scalar[k] = 0.0;
        }
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_vec, z_vec, 256 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_scalar, z_scalar, 256 * sizeof(double), cudaMemcpyHostToDevice);

        // vecs for averaging
        std::vector<float> log_dot_v1;
        std::vector<float> log_dot_v2;

        for (int average_count = 0; average_count < 3; average_count++)
        {
/*
            //  dot operation via 2 kernels
            cudaDeviceSynchronize();
            timer.reset();
            dot_s1<<<256, 256>>>(N, d_x, d_y, d_z_scalar);
            dot_s2<<<1, 256>>>(d_z_scalar);
            cudaMemcpy(z_scalar, d_z_scalar, 256 * sizeof(double), cudaMemcpyDeviceToHost); // copy data back (implicit synchronization point)
            cudaDeviceSynchronize();
            float elapsed_time_dot_v1 = timer.get();
            log_dot_v1.push_back(elapsed_time_dot_v1);
            //std::cout << "dotp1:" << z_scalar[0] << "\n";
*/

            // dot operation with GPU/CPU cooperation
            cudaDeviceSynchronize();
            timer.reset();
            dot_s1<<<256, 256>>>(N, d_x, d_y, d_z_vec);
            cudaMemcpy(z_vec, d_z_vec, 256 * sizeof(double), cudaMemcpyDeviceToHost); // copy data back (implicit synchronization point)
            double z_scalar_2 = std::accumulate(z_vec, z_vec + 256, 0.0);
            cudaDeviceSynchronize();
            float elapsed_time_dot_v2 = timer.get();
            log_dot_v2.push_back(elapsed_time_dot_v2);
            //std::cout << "dotp2:" << z_scalar_2 << "\n";
        }

        // build averages
        //float log_dot1_av = std::accumulate(log_dot_v2.begin(), log_dot_v2.end(), 0.0 / log_dot_v2.size());
        float log_dot2_av = std::accumulate(log_dot_v2.begin(), log_dot_v2.end(), 0.0 / log_dot_v2.size());

        // output
        std::cout << N << " " << log_dot2_av << std::endl;

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z_vec);
        cudaFree(d_z_scalar);
        free(x);
        free(y);
        free(z_scalar);
        free(z_vec);
    }

    return EXIT_SUCCESS;
}