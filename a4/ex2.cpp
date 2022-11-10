#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>

// a) implements 8 dot-product operations in one
__global__ void cuda_mdot_product(int N, double *x, double *y, double *results)
{
    __shared__ double shared_mem1[256]; // remember to only use 256 threads per block then!
    __shared__ double shared_mem2[256];
    __shared__ double shared_mem3[256];
    __shared__ double shared_mem4[256];
    __shared__ double shared_mem5[256];
    __shared__ double shared_mem6[256];
    __shared__ double shared_mem7[256];
    __shared__ double shared_mem8[256];

    double dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0, dot5 = 0, dot6 = 0, dot7 = 0, dot8 = 0;
    double w;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        w = x[i];
        dot1 += w * y[i];
        dot2 += w * y[i + N];
        dot3 += w * y[i + 2 * N];
        dot4 += w * y[i + 3 * N];
        dot5 += w * y[i + 4 * N];
        dot6 += w * y[i + 5 * N];
        dot7 += w * y[i + 6 * N];
        dot8 += w * y[i + 7 * N];
    }

    shared_mem1[threadIdx.x] = dot1;
    shared_mem2[threadIdx.x] = dot2;
    shared_mem3[threadIdx.x] = dot3;
    shared_mem4[threadIdx.x] = dot4;
    shared_mem5[threadIdx.x] = dot5;
    shared_mem6[threadIdx.x] = dot6;
    shared_mem7[threadIdx.x] = dot7;
    shared_mem8[threadIdx.x] = dot8;

    for (int k = blockDim.x / 2; k > 0; k /= 2)
    {
        __syncthreads();
        if (threadIdx.x < k)
        {
            shared_mem1[threadIdx.x] += shared_mem1[threadIdx.x + k];
            shared_mem2[threadIdx.x] += shared_mem2[threadIdx.x + k];
            shared_mem3[threadIdx.x] += shared_mem3[threadIdx.x + k];
            shared_mem4[threadIdx.x] += shared_mem4[threadIdx.x + k];
            shared_mem5[threadIdx.x] += shared_mem5[threadIdx.x + k];
            shared_mem6[threadIdx.x] += shared_mem6[threadIdx.x + k];
            shared_mem7[threadIdx.x] += shared_mem7[threadIdx.x + k];
            shared_mem8[threadIdx.x] += shared_mem8[threadIdx.x + k];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&results[0], shared_mem1[0]);
        atomicAdd(&results[1], shared_mem2[0]);
        atomicAdd(&results[2], shared_mem3[0]);
        atomicAdd(&results[3], shared_mem4[0]);
        atomicAdd(&results[4], shared_mem5[0]);
        atomicAdd(&results[5], shared_mem6[0]);
        atomicAdd(&results[6], shared_mem7[0]);
        atomicAdd(&results[7], shared_mem8[0]);
    }
}

// b) wraps the 8-dot-product kernel such that it can handle K = multiple of 8
void multiple_batches(int N, int K, double *x, double **y, double **results)
{
    int h = K / 8;
    for (; h > 0; h--)
    {
        cuda_mdot_product<<<256, 256>>>(N, x, y[h - 1], results[h - 1]);
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
    // Initialize CUBLAS:
    cublasHandle_t h;
    cublasCreate(&h);

    // outer loop over vector lengths
    for (int N = 1000; N < 1000001; N *= 10)
    {
        // inner loop over number of dot-products
        for (size_t K = 8; K < 33; K += 8)
        {
            // allocate host memory:
            double *x = (double *)malloc(sizeof(double) * N);
            double *results = (double *)malloc(sizeof(double) * K);
            double *results_cublas = (double *)malloc(sizeof(double) * K);
            double **v = (double **)malloc(sizeof(double *) * K / 8);
            for (size_t i = 0; i < K / 8; ++i)
            {
                v[i] = (double *)malloc(sizeof(double) * N * 8);
            }
            double **results3 = (double **)malloc(sizeof(double) * K / 8);
            for (size_t i = 0; i < K / 8; ++i)
            {
                results3[i] = (double *)malloc(sizeof(double) * 8);
            }
            double **cublas_v = (double **)malloc(sizeof(double *) * K);
            for (size_t i = 0; i < K; ++i)
            {
                cublas_v[i] = (double *)malloc(sizeof(double) * N);
            }

            // fill host arrays with values
            std::fill(x, x + N, 1.0);
            for (size_t i = 0; i < K / 8; ++i)
            {
                for (size_t j = 0; j < N * 8; ++j)
                {
                    v[i][j] = 1 + rand() / (1.1 * RAND_MAX);
                }
            }
            for (size_t i = 0; i < K; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    cublas_v[i][j] = 1 + rand() / (1.1 * RAND_MAX);
                }
            }

            // allocate device memory
            double *cuda_x;
            cudaMalloc((&cuda_x), sizeof(double) * N);
            double **cuda_v = (double **)malloc(sizeof(double *) * K / 8); // storing CUDA pointers on host!
            for (size_t i = 0; i < K / 8; ++i)
            {
                cudaMalloc((void **)(&cuda_v[i]), sizeof(double) * N * 8);
            }
            double **cuda_results3 = (double **)malloc(sizeof(double *) * K / 8); // storing CUDA pointers on host!
            for (size_t i = 0; i < K / 8; ++i)
            {
                cudaMalloc((void **)(&cuda_results3[i]), sizeof(double) * 8);
            }
            double **cuda_cublas_v = (double **)malloc(sizeof(double *) * K); // storing CUDA pointers on host!
            for (size_t i = 0; i < K; ++i)
            {
                cudaMalloc((void **)(&cuda_cublas_v[i]), sizeof(double) * N);
            }

            // Copy data to GPU
            cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
            for (size_t i = 0; i < K / 8; ++i)
            {
                cudaMemcpy(cuda_v[i], v[i], sizeof(double) * N * 8, cudaMemcpyHostToDevice);
            }
            for (size_t i = 0; i < K; ++i)
            {
                cudaMemcpy(cuda_cublas_v[i], cublas_v[i], sizeof(double) * N, cudaMemcpyHostToDevice);
            }

            // save data struc
            std::vector<float> log_time_custom;
            std::vector<float> log_time_cublas;

            // load timer
            Timer timer;

            // do operation a number of times to correct for singular effects
            for (int i = 0; i < 11; i++)
            {
                // custom
                CUDA_ERRCHK(cudaDeviceSynchronize());
                timer.reset();
                multiple_batches(N, K, cuda_x, cuda_v, cuda_results3);
                CUDA_ERRCHK(cudaDeviceSynchronize());
                float elapsed_time_0 = timer.get();
                if (i > 0) // during first run GPU has to warm up
                {
                    log_time_custom.push_back(elapsed_time_0);
                }

                // CUBLAS
                CUDA_ERRCHK(cudaDeviceSynchronize());
                timer.reset();
                for (size_t l = 0; l < K; ++l)
                {
                    cublasDdot(h, N, cuda_x, 1, cuda_cublas_v[l], 1, results_cublas + l);
                }
                CUDA_ERRCHK(cudaDeviceSynchronize());
                float elapsed_time_1 = timer.get();
                if (i > 0) // during first run GPU has to warm up
                {
                    log_time_cublas.push_back(elapsed_time_1);
                }
            }

            // output time average
            std::cout << N << " " << K << " " << build_av(log_time_custom) << " " << build_av(log_time_cublas) << std::endl;

            // Copy data to Host
            for (size_t i = 0; i < K / 8; ++i)
            {
                cudaMemcpy(results3[i], cuda_results3[i], sizeof(double) * 8, cudaMemcpyDeviceToHost);
            }

            /*
                // Reference calculation on CPU:
                for (size_t i=0; i<K/8; ++i) {
                results[k] = 0;
                for (size_t j=0; j<N*8; ++j) {
                    results[i] += x[j] * v[i][j];
                }
                }

                // Compare results
                std::cout << "Copying results back to host..." << std::endl;
                for (size_t i=0; i<K/8; ++i) {
                    for (size_t j=0; j<8; ++j) {
                        std::cout << results[i*8+j] << " on CPU " << results3[i][j] << " on CUDA. " << std::endl;
                    }
                }
            */

            // Clean up:
            free(results);
            free(results_cublas);
            free(x);
            cudaFree(cuda_x);
            for (size_t i = 0; i < K / 8; ++i)
            {
                free(v[i]);
                free(results3[i]);
                cudaFree(cuda_v[i]);
                cudaFree(cuda_results3[i]);
            }
            for (size_t i = 0; i < K; ++i)
            {
                free(cublas_v[i]);
                cudaFree(cuda_cublas_v[i]);
            }
            free(v);
            free(results3);
            free(cublas_v);
            cudaFree(cuda_v);
            cudaFree(cuda_results3);
            cudaFree(cuda_cublas_v);
        }
    }
    cublasDestroy(h);
    return EXIT_SUCCESS;
}