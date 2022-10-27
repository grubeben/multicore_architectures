#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "timer.hpp"
#include "cuda_errchk.hpp" // for error checking of CUDA calls

__global__ void vanilla_transpose(double *A, double *B, int N)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    int row_idx = t_idx / N;
    int col_idx = t_idx % N;

    while (row_idx < N)
    {
        int to = row_idx * N + col_idx;
        int from = col_idx * N + row_idx;
        // printf("th_id: %d, [%d,%d]  \n", t_idx, row_idx, col_idx);
        if (to != from) // diagonal is constant
        {
            B[to] = A[from];
        }
        row_idx += (int)((col_idx + total_threads) / N);
        col_idx = (col_idx + total_threads % N) % N; // col_idx = ((col_idx + total_threads) % N;
    }
}

__global__ void stride_optimal_attempt_transpose(double *A, double *B, int N)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    int elements_per_thread = N * N / total_threads;

    // printf("elements per thread: %d\n", elements_per_thread);

    int row_idx = (t_idx * elements_per_thread) / N;
    int col_idx = (t_idx * elements_per_thread) % N;
    int col_idx_old;
    int row_idx_old;

    while (row_idx < N)
    {
        // printf("UPDATE th_id: %d, [%d,%d]  \n", t_idx, row_idx, col_idx);
        for (int j = 0; j < elements_per_thread; j++)
        {
            int to = row_idx * N + col_idx;
            int from = col_idx * N + row_idx;
            if (to != from) // diagonal is constant
            {
                B[to] = A[from];
            }
            if (row_idx < N)
            {
                // store old values
                col_idx_old = col_idx;
                row_idx_old = row_idx;
                // printf("th_id: %d, [%d,%d]  \n", t_idx, row_idx, col_idx);
                row_idx = (int)row_idx + (col_idx + 1) / N;
                col_idx = ((col_idx + 1) % N);
            }
            else
            {
                j = elements_per_thread - 1;
            }
        }
        // correct for inner loop
        row_idx = row_idx_old;
        col_idx = (col_idx_old + (elements_per_thread * (total_threads - 1)) % N) % N;
        row_idx += ((col_idx_old + elements_per_thread * (total_threads - 1)) / N);
    }
}

__global__ void in_place_transpose(double *A, int N)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    int row_idx = t_idx / N;
    int col_idx = t_idx % N;

    while (row_idx < (N - 1) && col_idx < N && row_idx < col_idx) // A[N-1,N] is the last entry we change
    {
        int to = row_idx * N + col_idx;
        int from = col_idx * N + row_idx;

        if (from != to) // daigonal is constant
        {
            int temp = A[to];
            A[to] = A[from];
            A[from] = temp;
        }
        printf("th_id: %d, [%d,%d]  \n", t_idx, row_idx, col_idx);
        int row_idx_old = row_idx;
        row_idx += (int)((col_idx + total_threads) / (N - row_idx));
        col_idx = (((col_idx + total_threads) % N) + row_idx);
        printf("AFTER th_id: %d, [%d,%d]  \n", t_idx, row_idx, col_idx);
    }
}

void print_A(double *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << A[i * N + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(void)
{

    int N = 5;

    double *A, *cuda_A, *B, *cuda_B, *C, *cuda_C;
    Timer timer;

    // Allocate host memory and initialize
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++)
    {
        A[i] = i;
        B[i] = i;
        C[i] = i;
    }

    // Allocate device memory and copy host data over
    CUDA_ERRCHK(cudaMalloc(&cuda_A, N * N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_B, N * N * sizeof(double)));
    CUDA_ERRCHK(cudaMalloc(&cuda_C, N * N * sizeof(double)));

    // copy data over
    CUDA_ERRCHK(cudaMemcpy(cuda_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cuda_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // save data struc
    std::vector<float> log_vanilla_time;
    std::vector<float> log_stride_optimal_attempt_time;
    std::vector<float> log_in_place_time;

    print_A(A, N);

    // do operation a number of times to correct for singular effects
    for (int i = 0; i < 1; i++)
    {

        // vanilla_transpose
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        // Perform the vanilla_transpose operation
        // vanilla_transpose<<<(N + 255) / 256, 256>>>(cuda_A, cuda_B, N);
        // vanilla_transpose<<<1, 2>>>(cuda_A, cuda_B, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_vanilla_time.push_back(elapsed_time);
        }

        // stride_optimal_attempt_transpose
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        // Perform the vanilla_transpose operation
        // vanilla_transpose<<<(N + 255) / 256, 256>>>(cuda_A, cuda_C, N);
        // stride_optimal_attempt_transpose<<<2, 2>>>(cuda_A, cuda_C, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time3 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_stride_optimal_attempt_time.push_back(elapsed_time3);
        }

        // in_place_transpose
        CUDA_ERRCHK(cudaDeviceSynchronize());
        timer.reset();
        // Perform the vanilla_transpose operation
        in_place_transpose<<<1, 2>>>(cuda_A, N);
        CUDA_ERRCHK(cudaDeviceSynchronize());
        float elapsed_time2 = timer.get();
        if (i > 0) // during first run GPU has to warm up
        {
            log_in_place_time.push_back(elapsed_time2);
        }
    }

    // define averages
    float log_vanilla_time_av = std::accumulate(log_vanilla_time.begin(), log_vanilla_time.end(), 0.0);
    log_vanilla_time_av /= log_vanilla_time.size();
    float log_stride_optimal_attempt_time_av = std::accumulate(log_stride_optimal_attempt_time.begin(), log_stride_optimal_attempt_time.end(), 0.0);
    log_stride_optimal_attempt_time_av /= log_stride_optimal_attempt_time.size();
    float log_in_place_time_av = std::accumulate(log_in_place_time.begin(), log_in_place_time.end(), 0.0);
    log_in_place_time_av /= log_in_place_time.size();

    // output averages
    std::cout << log_vanilla_time_av << " " << ((2 * N * N - N) * sizeof(double)) / log_vanilla_time_av * 1e-9 << " GB/sec" << std::endl;
    std::cout << log_stride_optimal_attempt_time_av << " " << ((2 * N * N - N) * sizeof(double)) / log_stride_optimal_attempt_time_av * 1e-9 << " GB/sec" << std::endl;
    // std::cout << log_in_place_time_av << " " << (2 * N * N * sizeof(double)) / log_in_place_time_av * 1e-9 << " GB/sec" << std::endl;
    std::cout << std::endl;

    // copy data back (implicit synchronization point)
    CUDA_ERRCHK(cudaMemcpy(A, cuda_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpy(B, cuda_B, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_ERRCHK(cudaMemcpy(C, cuda_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    print_A(A, N);
    // print_A(B, N);
    // print_A(C, N);

    // missing free-statements ==> 800 bytes leak(N*N*sizeof(double)=10*10*8)
    free(A);
    free(B);
    free(C);
    CUDA_ERRCHK(cudaFree(cuda_A));
    CUDA_ERRCHK(cudaFree(cuda_B));
    CUDA_ERRCHK(cudaFree(cuda_C));

    CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work

    return EXIT_SUCCESS;
}
