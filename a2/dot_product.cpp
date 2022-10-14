#include <stdio.h>
// included in online evironment
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

__global__ void dot_v1(int n, double *x, double *y, double *z_scalar)
{
    // const int total_threads = blockDim.x * gridDim.x;     // number of threads
    int id = blockIdx.x * blockDim.x + threadIdx.x; // #block    #threads/block   #offset within block
    __shared__ double temp[65000];                  // shared temporal memory on GPU RAM vs. extern __shared__: this is used in case size of temp is unclear

    int opb = n / gridDim.x; // how many vector-entries does each block work on?
    if (n % gridDim.x != 0)
    {
        opb = n / gridDim.x + 1;
    }
    else
    {
        opb = n / gridDim.x;
    }

    // each block operates in certain range of vector (with size n)
    int lower_bound = blockIdx.x * opb;
    int upper_bound = lower_bound + opb;
    if (upper_bound < n)
    {
        upper_bound = n;
    }

    // partial multiplication
    for (int i = lower_bound + threadIdx.x; i < upper_bound; i += blockDim.x)
    {
        temp[id] = x[i] * y[i];
    }

    __syncthreads(); // sync threads
    // is this where kernel 2 starts?

    // parallel primitives - stride
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            temp[threadIdx.x] += temp[threadIdx.x + stride];
    }

    // only one thread writes results
    if (threadIdx.x == 0)
    {
        z_scalar[0] = temp[0];
    }
}

__global__ void dot_v2(int n, double *x, double *y, double *z_vec)
{
    // const int total_threads = blockDim.x * gridDim.x; // number of threads
    int id = blockIdx.x * blockDim.x + threadIdx.x; // #block    #threads/block   #offset within block
    __shared__ double temp[65000];                  // shared temporal memory on GPU RAM vs. extern __shared__: this is used in case size of temp is unclear

    int opb = n / gridDim.x; // how many vector-entries does each block work on?
    if (n % gridDim.x != 0)
    {
        opb = n / gridDim.x + 1;
    }
    else
    {
        opb = n / gridDim.x;
    }

    // each block operates in certain range of vector (with size n)
    int lower_bound = blockIdx.x * opb;
    int upper_bound = lower_bound + opb;
    if (upper_bound < n)
    {
        upper_bound = n;
    }

    // partial multiplication
    for (int i = lower_bound + threadIdx.x; i < upper_bound; i += blockDim.x)
    {
        temp[id] = x[i] * y[i];
    }

    // only one thread writes results, so that we can copy them to CPU
    if (threadIdx.x == 0)
    {
        z_vec = temp;
    }
}

int main(int argc, char **argv)
{
    std::vector<int> n = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000}; // initialize vector lengths
    Timer timer;                                                                                 // load timer

    for (int j = 0; j < n.size(); j++) // outer loop over vector lengths
    {
        int N = n[j];

        // allocate device memory and initiate via copy
        double *x, *d_x, *y, *d_y, *z_vec, *d_z_vec, *z_scalar, *d_z_scalar, *z_scalar_2;
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        z_vec = (double *)malloc(N * sizeof(double));
        z_scalar = (double *)malloc(N * sizeof(double));
        z_scalar_2 = (double *)malloc(N * sizeof(double));
        cudaMalloc(&d_x, N * sizeof(double));
        cudaMalloc(&d_y, N * sizeof(double));
        cudaMalloc(&d_z_vec, N * sizeof(double));
        cudaMalloc(&d_z_scalar, sizeof(double));

        for (int k = 0; k < N; k++)
        {
            x[k] = 1.0;
            y[k] = 1.0;
            z_vec[k] = 0.0;
            z_scalar[k] = 0.0;
            z_scalar_2[k] = 0.0;
        }
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_vec, z_vec, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_scalar, z_scalar, N * sizeof(double), cudaMemcpyHostToDevice);

        // vecs for averaging
        std::vector<float> log_dot_v1;
        std::vector<float> log_dot_v2;

        for (int i = 0; i < 10; i++)
        {
            int n_threads = 256;
            int n_blocks = (N + n_threads - 1) / n_threads;
            
            //  dot_v1 operation
            cudaDeviceSynchronize();
            timer.reset();
            dot_v1<<<n_blocks, n_threads>>>(N, d_x, d_y, d_z_scalar);
            cudaMemcpy(z_scalar, d_z_scalar, sizeof(double), cudaMemcpyDeviceToHost); // copy data back (implicit synchronization point)
            cudaDeviceSynchronize();
            float elapsed_time_dot_v1 = timer.get();
            log_dot_v1.push_back(elapsed_time_dot_v1);

            // dot_v2 operation
            cudaDeviceSynchronize();
            timer.reset();
            dot_v2<<<n_blocks, n_threads>>>(N, d_x, d_y, d_z_vec);
            cudaMemcpy(z_vec, d_z_vec, N * sizeof(double), cudaMemcpyDeviceToHost); // copy data back (implicit synchronization point)
            int z_vec_len = sizeof(z_vec) / sizeof(z_vec[0]);                       // convert [] to vec in order to use accumalte
            std::vector<double> dest(z_vec, z_vec + z_vec_len);
            double z_scalar_2 = std::accumulate(dest.begin(), dest.end(), 0.0 / dest.size());
            cudaDeviceSynchronize();
            float elapsed_time_dot_v2 = timer.get();
            log_dot_v2.push_back(elapsed_time_dot_v2);
        }

        // build averages
        float log_dot1_av = std::accumulate(log_dot_v2.begin(), log_dot_v2.end(), 0.0 / log_dot_v2.size());
        float log_dot2_av = std::accumulate(log_dot_v1.begin(), log_dot_v1.end(), 0.0 / log_dot_v1.size());

        // output
        std::cout << N << " " << log_dot1_av << " " << log_dot2_av << std::endl;

        free(x);
        free(y);
        free(z_scalar);
        free(z_scalar_2);
        free(z_vec);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z_vec);
        cudaFree(d_z_scalar);
    }

    return EXIT_SUCCESS;
}