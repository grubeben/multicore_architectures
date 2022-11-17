#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <numeric>
#include <algorithm>

// b) Kernel launch latency
__global__ void nullKernel() {}

// c) Practical peak memory bandwidth via vector addition
__global__ void vectorAddition(double *a, double *b, double *c, int n)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < n)
    {
        c[threadId] = a[threadId] + b[threadId];
    }
}

// d) atomicAdd() updates per second
__global__ void AtomicAdds(double *result, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        atomicAdd(result, 1.0);
    }
}

// e) Peak floating point rate per vector triad
__global__ void vectorTriad(double *a, double *b, double *c, int n)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < n)
    {
        c[threadId] += a[threadId] * b[threadId];
    }
}

// compute averages
float build_av(std::vector<float> log_vec)
{
    float av = std::accumulate(log_vec.begin(), log_vec.end(), 0.0);
    av /= log_vec.size();
    return av;
}

int main(void)
{
    int N = 1;
    int n = 10;
    double *x, *cuda_x, *y, *cuda_y, *z, *cuda_z, *xn, *cuda_xn;

    // Allocate host memory and initialize
    x = (double *)malloc(N * sizeof(double));
    xn = (double *)malloc(n * sizeof(double));
    y = (double *)malloc(N * sizeof(double));
    z = (double *)malloc(N * sizeof(double));

    std::fill(x, x + N, 1);
    std::fill(xn, xn + n, 1);
    std::fill(y, y + N, 1);
    std::fill(z, z + N, 0);

    // Allocate device memory
    CUDA_ERRCHK(cudaMalloc(&cuda_x, N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_xn, n * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_y, N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_z, N * sizeof(double)));

    // copy data over
    CUDA_ERRCHK(cudaMemcpy(cuda_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_xn, xn, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_y, y, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_z, z, N * sizeof(double), cudaMemcpyHostToDevice));

    // save data struc
    std::vector<float> log_a;
    std::vector<float> log_b;
    std::vector<float> log_c;
    std::vector<float> log_d;
    std::vector<float> log_e;

    // initiate timer
    Timer timer;

    // do operation a number of times to correct for singular effects
    for (int i = 0; i < 100; i++)
    {
        // a) PCI express
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        CUDA_ERRCHK(cudaMemcpy(cuda_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_1_element = timer.get();

        timer.reset();
        CUDA_ERRCHK(cudaMemcpy(cuda_xn, xn, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_n_element = timer.get();

        if (i > 0) // during first run GPU has to warm up
        {
            log_a.push_back(elapsed_time_1_element - (elapsed_time_n_element - elapsed_time_1_element) / (n-1));
        }

        // b) kernel launch
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        nullKernel<<<256, 256>>>();
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_1 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_b.push_back(elapsed_time_1);
        }

        // c) peak memory bandwidth
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        vectorAddition<<<256, 256>>>(cuda_z, cuda_x, cuda_y, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_2 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {

            log_c.push_back(3 * N * sizeof(double) / (1e9 * elapsed_time_2));
        }

        // d) atomicAdd()/ sec
        const int number_adds = 100;
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        AtomicAdds<<<256, 256>>>(cuda_x, number_adds);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_3 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_d.push_back(number_adds / elapsed_time_3);
        }

        // e) peak floating point rate
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        vectorTriad<<<256, 256>>>(cuda_x, cuda_y, cuda_z, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_4 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            float rp_rate = 2 * N / (1e9 * elapsed_time_4);
            log_e.push_back(rp_rate);
        }
    }

    // define averages
    float log_a_av = build_av(log_a);
    float log_b_av = build_av(log_b);
    float log_c_av = build_av(log_c);
    float log_d_av = build_av(log_d);
    float log_e_av = build_av(log_e);

    // output
    std::cout << "PCI Express latency:                  " << 1e3 * log_a_av << "    [ms]" << std::endl;
    std::cout << "Kernel launch latency:                " << 1e3 * log_b_av << "    [ms]" << std::endl;
    std::cout << "Practical peak memory performance:    " << log_c_av << "  [GB/s]" << std::endl;
    std::cout << "Maximum # AtomicAdds():               " << log_d_av << std::endl;
    std::cout << "Peak floating point rate:             " << log_e_av << "  [GFLOPs/s]" << std::endl;

    free(x);
    free(xn);
    free(y);
    CUDA_ERRCHK(cudaFree(cuda_x));
    CUDA_ERRCHK(cudaFree(cuda_xn));
    CUDA_ERRCHK(cudaFree(cuda_y));
    CUDA_ERRCHK(cudaFree(cuda_z));

    CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work

    return EXIT_SUCCESS;
}
