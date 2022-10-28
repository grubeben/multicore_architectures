#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "timer.hpp"
#include "cuda_errchk.hpp" // for error checking of CUDA calls

// assume 2 threads: th_0 and th_1 access pattern for A.size()=NxN=9: 010101010
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

__global__ void vanilla_transpose_in_place(double *A, int N)
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
        if (to != from && row_idx < col_idx) // only access upper triangular entries
        {
            int temp = A[to];
            A[to] = A[from];
            A[from] = temp;
        }
        row_idx += (int)((col_idx + total_threads) / N);
        col_idx = (col_idx + total_threads % N) % N; // col_idx = ((col_idx + total_threads) % N;
    }
}

// assume 2 threads: th_0 and th_1 access pattern for A.size()=NxN=9: 0000 1111 0
__global__ void stride_optimal_attempt_transpose(double *A, double *B, int N)
{
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    int elements_per_thread = N * N / total_threads;
    // printf("elements per thread: %d\n", elements_per_thread);
    // define starting point for threads
    int row_idx = (t_idx * elements_per_thread) / N;
    int col_idx = (t_idx * elements_per_thread) % N;
    int col_idx_old;
    int row_idx_old;

    while (row_idx < N)
    {
        // active thread works on #elements_per_thread contiguous matrix entries
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
                // active thread moves on by one cell
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
        // update in case total_threads < N*N
        col_idx = (col_idx_old + (elements_per_thread * (total_threads - 1)) % N) % N;
        row_idx += ((col_idx_old + elements_per_thread * (total_threads - 1)) / N);
    }
}

// access only upper triangular elements by idx instead of if statement
__global__ void complicated_in_place_attempt_transpose(double *A, int N) // FAULTY !!
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

// inspired by https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void read_optimal_transpose(double *A, double *B, TILE_DIM = 32, int BLOCK_ROWS = 8)
{
    int id_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int id_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM; //#blocks*tile_dimension = N?

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int from = (id_y + j) * width + id_x;
        int to = id_x * width + (id_y + j);
        if (to != from) // diagonal is constant
        {
            B[to] = A[from];
        }
    }
}

// inspired by https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void read_write_optimal_transpose(double *A, double *B, TILE_DIM = 32, int BLOCK_ROWS = 8)
{
    __shared__ double tile[TILE_DIM][TILE_DIM + 1]; // introduce shared mem for sub matrix (tile) incl- padding

    int id_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int id_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM; // #blocks*tile_dimension = N?

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int from = (id_y + j) * width + id_x;
        tile[threadIdx.y + j][threadIdx.x] = A[from]; // load transposed into shared memory
    }
    __syncthreads();

    // redistribute indices
    int id_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int id_y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int to = (id_y + j) * width + id_x;
        B[to] = tile[threadIdx.x][threadIdx.y + j]; // write column of shared memory into output matrix
    }
}

// c) inspired by https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// this only works for multiples of N=16 starting with N=32
__global__ void read_write_optimal_in_place_transpose(double *A, TILE_DIM = 16, int BLOCK_ROWS = 16)
{
    __shared__ double tile_1[TILE_DIM][TILE_DIM + 1];
    __shared__ double tile_2[TILE_DIM][TILE_DIM + 1];

    int id_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int id_y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM; // #blocks*tile_dimension = N?

    if (blockIdx.x >= blockIdx.y)
    {
        int from = (id_y)*width + id_x;
        int to = id_x * width + (id_y + j);

        if (from != to)
        {

            tile_1[threadIdx.y][threadIdx.x] = A[from]; // load transposed into shared memory
            tile_2[threadIdx.y][threadIdx.x] = A[to];   // load transposed of "to be replaced tile" into shared memory
                                                        // the second read is NOT coalesced

            A[to] = tile_1[threadIdx.x][threadIdx.y + j]; // replace both tiles
            A[from] = tile_2[threadIdx.x][threadIdx.y + j];
        }
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

float build_av(std::vector<float> log_vec)
{
    float av = std::accumulate(log_vec.begin(), log_vec.end(), 0.0);
    av /= log_vec.size();
    return av;
}

int main(void)
{
    // outer loop over vector lengths
    // for (int N = 32; N < 256; N*=2) // benchmarking
    for (int N = 16; N < 17; N += 2) // test
    {

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
        std::vector<float> log_vanilla_in_place_time;
        std::vector<float> log_stride_optimal_attempt_time;
        std::vector<float> log_read_write_optimal_time;
        std::vector<float> log_in_place_time;

        print_A(A, N);

        // do operation a number of times to correct for singular effects
        for (int i = 0; i < 11; i++)
        {

            // vanilla_transpose
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            // Perform the vanilla_transpose operation
            // vanilla_transpose<<<(N + 255) / 256, 256>>>(cuda_A, cuda_B, N);
            vanilla_transpose<<<1, 2>>>(cuda_A, cuda_B, N);
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
            stride_optimal_attempt_transpose<<<(N + 255) / 256, 256>>>(cuda_A, cuda_C, N);
            CUDA_ERRCHK(cudaDeviceSynchronize());
            float elapsed_time3 = timer.get();
            if (i > 0) // during first run GPU has to warm up
            {
                log_stride_optimal_attempt_time.push_back(elapsed_time3);
            }

            // read_write_optimal_transpose
            CUDA_ERRCHK(cudaDeviceSynchronize());
            dim3 block(32, 8); // 32 threads per blocks in x-dimension, 8 threads per blocks in y-dimension
            dim3 grid(N / 32 + 1, N / 8 + 1);
            timer.reset();
            read_optimal_transpose<<<block, grid>>>(cuda_A, cuda_B, N);
            CUDA_ERRCHK(cudaDeviceSynchronize());
            float elapsed_time5 = timer.get();
            if (i > 0) // during first run GPU has to warm up
            {
                log_read_write_optimal_attempt_time.push_back(elapsed_time5);
            }

            // vanilla_transpose_in_place
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            // Perform the vanilla_transpose operation
            // vanilla_transpose<<<(N + 255) / 256, 256>>>(cuda_A, cuda_B, N);
            vanilla_transpose_in_place<<<1, 2>>>(cuda_A, N);
            CUDA_ERRCHK(cudaDeviceSynchronize());
            float elapsed_time4 = timer.get();
            if (i > 0) // during first run GPU has to warm up
            {
                log_vanilla_in_place_time.push_back(elapsed_time4);
            }

            // read_write_optimal_in_place_transpose
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            dim3 block(16, 16); // 16 threads per blocks in x-dimension, 16 threads per blocks in y-dimension
            dim3 grid(N / 16 + 1, N / 16 + 1);
            // read_write_optimal_in_place_transpose<<<1, 2>>>(cuda_A, N);
            CUDA_ERRCHK(cudaDeviceSynchronize());
            float elapsed_time6 = timer.get();
            if (i > 0) // during first run GPU has to warm up
            {
                log_in_place_time.push_back(elapsed_time6);
            }
        }

        // define averages
        float log_vanilla_time_av = build_av(log_vanilla_time);
        float log_vanilla_in_place_time_av = build_av(log_vanilla_in_place_time);
        float log_stride_optimal_attempt_time_av = build_av(log_stride_optimal_attempt_time);
        float log_in_place_time_av = build_av(log_in_place_time);
        float log_read_write_optimal_time_av = build_av(read_write_optimal_time);

        // output averages
        std::cout << N << " " << ((2 * N * N - N) * sizeof(double)) / log_vanilla_time_av * 1e-9 << " " << ((2 * N * N - N) * sizeof(double)) / log_vanilla_in_place_time_av * 1e-9 << " " << ((2 * N * N - N) * sizeof(double)) / log_stride_optimal_attempt_time_av * 1e-9 << " " << (2 * N * N * sizeof(double)) / log_read_write_optimal_time_av * 1e-9 << std::endl;
        // std::cout << log_in_place_time_av << " " << (2 * N * N * sizeof(double)) / log_in_place_time_av * 1e-9 << " GB/sec" << std::endl;
        std::cout << std::endl;

        // copy data back (implicit synchronization point)
        CUDA_ERRCHK(cudaMemcpy(A, cuda_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_ERRCHK(cudaMemcpy(B, cuda_B, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_ERRCHK(cudaMemcpy(C, cuda_C, N * N * sizeof(double), cudaMemcpyDeviceToHost));

        print_A(A, N);
        print_A(B, N);
        // print_A(C, N);

        // missing free-statements ==> 800 bytes leak(N*N*sizeof(double)=10*10*8)
        free(A);
        free(B);
        free(C);
        CUDA_ERRCHK(cudaFree(cuda_A));
        CUDA_ERRCHK(cudaFree(cuda_B));
        CUDA_ERRCHK(cudaFree(cuda_C));

        CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work
    }
    return EXIT_SUCCESS;
}