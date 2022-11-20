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

// e) Peak floating point rate per vector triad
__global__ void vectorTriad(double *aa, double *bb, double *cc, int n, int X)
{
    double a = aa[blockIdx.x * blockDim.x + threadIdx.x];
    double b = bb[blockIdx.x * blockDim.x + threadIdx.x];
    double c = cc[blockIdx.x * blockDim.x + threadIdx.x];

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < n)
    {
        for (int i = 0; i < X; i++)
        {
            // 12 times
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
            c += a * b;
        }
        aa[blockIdx.x * blockDim.x + threadIdx.x] += c;
    }
}

// compute averages
float build_av(std::vector<float> log_vec)
{
    float av = std::accumulate(log_vec.begin(), log_vec.end(), 0.000000);
    av /= log_vec.size();
    return av;
}

int main(void)
{
    int n_blocks = 1e4;
    int n_threads = 1024;
    int N = n_blocks * n_threads;
    int Na = 1;
    int n = 1e6;
    double *x, *cuda_x, *y, *cuda_y, *z, *cuda_z, *xn, *cuda_xn, *xa, *cuda_xa;

    // Allocate host memory and initialize
    x = (double *)malloc(N * sizeof(double));
    xa = (double *)malloc(Na * sizeof(double));
    xn = (double *)malloc(n * sizeof(double));
    y = (double *)malloc(N * sizeof(double));
    z = (double *)malloc(N * sizeof(double));

    std::fill(x, x + N, 1);
    std::fill(xa, xa + Na, 1);
    std::fill(xn, xn + n, 1);
    std::fill(y, y + N, 1);
    std::fill(z, z + N, 0);

    // Allocate device memory
    CUDA_ERRCHK(cudaMalloc(&cuda_x, N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_xa, Na * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_xn, n * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_y, N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_z, N * sizeof(double)));

    // copy data over
    CUDA_ERRCHK(cudaMemcpy(cuda_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_xa, xa, Na * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_xn, xn, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_y, y, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_z, z, N * sizeof(double), cudaMemcpyHostToDevice));

    // save data struc
    std::vector<float> log_a;
    std::vector<float> log_b;
    std::vector<float> log_c;
    std::vector<float> log_d;
    std::vector<double> log_e;

    // initiate timer
    Timer timer;

    // do operation a number of times to correct for singular effects
    for (int i = 0; i < 100; i++)
    {
        // a) PCI express (vs PCI Gen3 latency)
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        CUDA_ERRCHK(cudaMemcpy(cuda_xa, xa, Na * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_N_element = timer.get();

        timer.reset();
        CUDA_ERRCHK(cudaMemcpy(cuda_xn, xn, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_n_element = timer.get();

        if (i > 0) // during first run GPU has to warm up
        {
            log_a.push_back((elapsed_time_n_element * Na - elapsed_time_N_element * n) / (1 - n));
        }

        // b) kernel launch
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        nullKernel<<<1, 1>>>();
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_1 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_b.push_back(elapsed_time_1);
        }

        // c) peak memory bandwidth
        // RTX has 3584= 28*4*32 physical cores, we need to utilize all of them
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        vectorAddition<<<n_blocks, n_threads>>>(cuda_z, cuda_x, cuda_y, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time_2 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {

            log_c.push_back(3 * N * sizeof(double) / (1e9 * elapsed_time_2));
        }
    }
    for (int j = 0; j < 4; j++)
    {
        // e) peak floating point rate
        const int number_triads = 10000;
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        vectorTriad<<<1024, 1024>>>(cuda_x, cuda_y, cuda_z, 1024 * 1024, number_triads);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        double elapsed_time_4 = timer.get();
        if (j > 0) // during first run GPU has to warm up
        {
            double rp_rate = number_triads * 12 * 2. * 1024 * 1024 / (1e9 * elapsed_time_4);
            log_e.push_back(rp_rate);
        }
    }

    // define averages
    float log_a_av = build_av(log_a);
    float log_b_av = build_av(log_b);
    float log_c_av = build_av(log_c);
    float log_d_av = build_av(log_d);

    double log_e_av = 0;
    for (size_t m = 0; m < (size_t)log_e.size(); m++)
    {
        log_e_av += log_e[m];
    }
    log_e_av /= 1e3 * log_e.size();

    // output
    std::cout << "PCI Express latency:                  " << 1e6 * log_a_av << "    [micro-s]" << std::endl;
    std::cout << "Kernel launch latency:                " << 1e6 * log_b_av << "    [micro-s]" << std::endl;
    std::cout << "Practical peak memory performance:    " << log_c_av << "  [GB/s]" << std::endl;
    std::cout << "Maximum # AtomicAdds():               " << log_d_av << std::endl;
    std::cout << "Peak floating point rate:             " << log_e_av << "  [TFLOPs/s]" << std::endl;

    free(x);
    free(xa);
    free(xn);
    free(y);
    CUDA_ERRCHK(cudaFree(cuda_x));
    CUDA_ERRCHK(cudaFree(cuda_xa));
    CUDA_ERRCHK(cudaFree(cuda_xn));
    CUDA_ERRCHK(cudaFree(cuda_y));
    CUDA_ERRCHK(cudaFree(cuda_z));

    CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work

    return EXIT_SUCCESS;
}
