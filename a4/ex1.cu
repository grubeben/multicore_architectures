#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "timer.hpp"
#include "cuda_errchk.hpp" // for error checking of CUDA calls

// dot product as performance reference
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

// a) shared_memory kernel computing simple_sum, norm1, norm2 and number_zeros
__global__ void shared_mem(int N, double *x, double *results)
{
  __shared__ double shared_mem_ss[256];    // shared memory simple sum
  __shared__ double shared_mem_as[256];    // shared memory absolute sum
  __shared__ double shared_mem_2norm[256]; // shared memory 2nrom
  __shared__ double shared_mem_zeros[256]; // shared memory zeros

  // cover for N> total_threads
  double thread_sum = 0;
  double absolute_thread_sum = 0;
  double thread_2norm = 0;
  double thread_zeros = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    // access x[i] 4x? RTX3060 has a cache for every SM, does that help here?
    thread_sum += x[i];
    absolute_thread_sum += abs(x[i]);
    thread_2norm += pow(x[i], 2);
    if (x[i] == 0)
    {
      thread_zeros += 1;
    }
  }

  shared_mem_ss[threadIdx.x] = thread_sum;
  shared_mem_as[threadIdx.x] = absolute_thread_sum;
  shared_mem_2norm[threadIdx.x] = thread_2norm;
  shared_mem_zeros[threadIdx.x] = thread_zeros;
  for (int k = blockDim.x / 2; k > 0; k /= 2)
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      shared_mem_ss[threadIdx.x] += shared_mem_ss[threadIdx.x + k];
      shared_mem_as[threadIdx.x] += shared_mem_as[threadIdx.x + k];
      shared_mem_2norm[threadIdx.x] += shared_mem_2norm[threadIdx.x + k];
      shared_mem_zeros[threadIdx.x] += shared_mem_zeros[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(&results[0], shared_mem_ss[0]); // adds shared_mem[0] to results and returns results[0]
    atomicAdd(&results[1], shared_mem_as[0]);
    atomicAdd(&results[2], shared_mem_2norm[0]); // take root??
    atomicAdd(&results[3], shared_mem_zeros[0]);
  }
}

// b) fixed thread-number warf-shuffle kernel computing simple_sum, norm1, norm2 and number_zeros
__global__ void warp_shuffle(int N, double *x, double *results)
{
  int total_threads = blockDim.x * gridDim.x;

  // declare local variables
  double thread_sum = 0;
  double absolute_thread_sum = 0;
  double thread_2norm = 0;
  double thread_zeros = 0;

  // cover for N > total_threads
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += total_threads)
  {
    thread_sum += x[i];
    absolute_thread_sum += abs(x[i]);
    thread_2norm += pow(x[i], 2);
    if (x[i] == 0)
    {
      thread_zeros += 1;
    }
  }

  // we now have total_threads - threads that need to be reduced; 32 per warp
  for (int i = warpSize / 2; i > 0; i = i / 2)
  {
    // __shfl_down_sync(all threads take part, loval var, how far too pass down local var);
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, i);
    absolute_thread_sum += __shfl_down_sync(0xFFFFFFFF, absolute_thread_sum, i);
    thread_2norm += __shfl_down_sync(0xFFFFFFFF, thread_2norm, i);
    thread_zeros += __shfl_down_sync(0xFFFFFFFF, thread_zeros, i);
  }

  if (threadIdx.x % 32 == 0)
  {
    atomicAdd(&results[0], thread_sum);
    atomicAdd(&results[1], absolute_thread_sum);
    atomicAdd(&results[2], thread_2norm); // take root??
    atomicAdd(&results[3], thread_zeros);
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
  // outer loop over vector lengths
  for (int N = 10; N < 10 ^ 8; N *= 10) // benchmarking
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
