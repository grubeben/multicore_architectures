#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "timer.hpp"
#include "cuda_errchk.hpp" // for error checking of CUDA calls



// dot-product <x,ki> for i= 0..7
__global__ void k8_dot_product(int N, double *x, double **y, double *results)
{
    

}


//reference dot-product
__global__ void cuda_dot_product(int N, double *x, double *y, double *alpha)
{
  __shared__ double shared_mem[512];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2)
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0)
    atomicAdd(alpha, shared_mem[0]);
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
  // outer loop over vector lengths
  for (int N = 10; N < 10^8; N *= 10) // benchmarking
  // for (int N = 32; N < 33; N += 2) // test
  {

    double *x, *cuda_x, *results, *cuda_results, alpha, *cuda_alpha;

    // Allocate host memory and initialize
    x = (double *)malloc(N * sizeof(double));
    results = (double *)malloc(4 * sizeof(double));
    alpha = 0;

    std::fill(x, x + N, 1);
    std::fill(results, results + 4, 0);

    // Allocate device memory and copy host data over
    CUDA_ERRCHK(cudaMalloc(&cuda_x, N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_results, 4 * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double)));

    // copy data over
    CUDA_ERRCHK(cudaMemcpy(cuda_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_results, results, 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));

    // save data struc
    std::vector<float> log_shared_mem;
    std::vector<float> log_ws_fixed;
    std::vector<float> log_ws_elastic;
    std::vector<float> log_dot_product;

    Timer timer;

    // do operation a number of times to correct for singular effects
    for (int i = 0; i < 11; i++)
    {
      // reference performance dot_product
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();
      cuda_dot_product<<<512, 512>>>(N, cuda_x, cuda_x, cuda_alpha);
      CUDA_ERRCHK(cudaDeviceSynchronize());
      float elapsed_time_0 = timer.get();
      if (i > 0) // during first run GPU has to warm up
      {
        log_dot_product.push_back(elapsed_time_0);
      }

      // shared_memory implementation
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();
      shared_mem<<<256, 256>>>(N, cuda_x, cuda_results);
      CUDA_ERRCHK(cudaDeviceSynchronize());
      float elapsed_time_1 = timer.get();
      if (i > 0) // during first run GPU has to warm up
      {
        log_shared_mem.push_back(elapsed_time_1);
      }

      // warp shuffle only inmplementation (with fixed number of threads)
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();
      warp_shuffle<<<256, 256>>>(N, cuda_x, cuda_results);
      CUDA_ERRCHK(cudaDeviceSynchronize());
      float elapsed_time_2 = timer.get();
      if (i > 0) // during first run GPU has to warm up
      {
        log_ws_fixed.push_back(elapsed_time_2);
      }

      // warp shuffle only inmplementation (with number of threads as function of N)
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();
      warp_shuffle<<<(N + 256) / 256, 256>>>(N, cuda_x, cuda_results);
      CUDA_ERRCHK(cudaDeviceSynchronize());
      float elapsed_time_3 = timer.get();
      if (i > 0) // during first run GPU has to warm up
      {
        log_ws_elastic.push_back(elapsed_time_3);
      }
    }

    // define averages
    float log_shared_mem_av = build_av(log_shared_mem);
    float log_ws_fixed_av = build_av(log_ws_fixed);
    float log_ws_elastic_av = build_av(log_ws_elastic);
    float log_dot_product_av = build_av(log_dot_product);

    // output time averages
    std::cout << N << " " << log_dot_product_av << " " << log_shared_mem_av << " " << log_ws_fixed_av << " " << log_ws_elastic_av << std::endl;

    // output bandwidth averages
    // std::cout << N << " " << ((2 * N * N - N) * sizeof(double)) / log_dot_product_av * 1e-9 << " " << ((2 * N * N - N) * sizeof(double)) / log_shared_mem_av * 1e-9 << " " << ((2 * N * N - N) * sizeof(double)) / log_ws_fixed_av*1e-9 << " " << ((2 * N * N - N) * sizeof(double)) / log_ws_elastic_av * 1e-9 << std::endl;
    // std::cout << std::endl;

    // copy data back (implicit synchronization point)
    // CUDA_ERRCHK(cudaMemcpy(x, cuda_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpy(results, cuda_results, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));

    free(x);
    free(results);
    CUDA_ERRCHK(cudaFree(cuda_x));
    CUDA_ERRCHK(cudaFree(cuda_results));
    CUDA_ERRCHK(cudaFree(cuda_alpha));

    CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work
  }
  return EXIT_SUCCESS;
}
