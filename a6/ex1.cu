#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "cuda_errchk.hpp"

__global__ void scan_kernel_1(double const *X,
                              double *Y,
                              int N,
                              double *carries)
{
    __shared__ double shared_buffer[256];
    double my_value;

    unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
    unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
    unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);
    unsigned int block_offset = 0;

    // run scan on each section
    for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    {
        // load data:
        my_value = (i < N) ? X[i] : 0; // conditional operator if(i<N){my_value=X[i];}else{my_value=0}

        // inclusive scan in shared buffer:
        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
        {
            __syncthreads();
            shared_buffer[threadIdx.x] = my_value;
            __syncthreads();
            if (threadIdx.x >= stride)
                my_value += shared_buffer[threadIdx.x - stride];
        }
        __syncthreads();
        shared_buffer[threadIdx.x] = my_value;
        __syncthreads();

        // exclusive scan requires us to write a zero value at the beginning of each block
        my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

        // write to output array
        if (i < N)
            Y[i] = block_offset + my_value;

        block_offset += shared_buffer[blockDim.x - 1];
    }

    // write carry:
    if (threadIdx.x == 0)
        carries[blockIdx.x] = block_offset;
}
__global__ void scan_kernel_1_i(double const *X,
                                double *Y,
                                int N,
                                double *carries)
{
    __shared__ double shared_buffer[256];
    double my_value;

    unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
    unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
    unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);
    unsigned int block_offset = 0;

    // run scan on each section
    for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    {
        // load data:
        my_value = (i < N) ? X[i] : 0; // conditional operator if(i<N){my_value=X[i];}else{my_value=0}

        // inclusive scan in shared buffer:
        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
        {
            __syncthreads();
            shared_buffer[threadIdx.x] = my_value;
            __syncthreads();
            if (threadIdx.x >= stride)
                my_value += shared_buffer[threadIdx.x - stride];
        }
        __syncthreads();
        shared_buffer[threadIdx.x] = my_value;
        __syncthreads();

        // exclusive scan requires us to write a zero value at the beginning of each block
        // my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
        my_value = shared_buffer[threadIdx.x];

        // write to output array
        if (i < N)
            Y[i] = block_offset + my_value;

        block_offset += shared_buffer[blockDim.x - 1];
    }

    // write carry:
    if (threadIdx.x == 0)
        carries[blockIdx.x] = block_offset;
}

// exclusive-scan of carries
__global__ void scan_kernel_2(double *carries)
{
    __shared__ double shared_buffer[256];

    // load data:
    double my_carry = carries[threadIdx.x];

    // exclusive scan in shared buffer:

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        shared_buffer[threadIdx.x] = my_carry;
        __syncthreads();
        if (threadIdx.x >= stride)
            my_carry += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();

    // write to output array
    carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}

__global__ void scan_kernel_3(double *Y, int N,
                              double const *carries)
{
    unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
    unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
    unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);

    __shared__ double shared_offset;

    if (threadIdx.x == 0)
        shared_offset = carries[blockIdx.x];

    __syncthreads();

    // add offset to each element in the block:
    for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
        if (i < N)
            Y[i] += shared_offset;
}

void exclusive_scan(double const *input,
                    double *output, int N)
{
    int num_blocks = 256;
    int threads_per_block = 256;

    double *carries;
    cudaMalloc(&carries, sizeof(double) * num_blocks);

    // First step: Scan within each thread group and write carries
    scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);

    // Second step: Compute offset for each thread group (exclusive scan for each thread group)
    scan_kernel_2<<<1, num_blocks>>>(carries);

    // Third step: Offset each thread group accordingly
    scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);

    cudaFree(carries);
}

// c) inclusive scan b modifying exclusive
void inclusive_scan(double const *input,
                    double *output, int N)
{
    int num_blocks = 256;
    int threads_per_block = 256;

    double *carries;
    cudaMalloc(&carries, sizeof(double) * num_blocks);

    // First step: Scan within each thread group and write carries
    scan_kernel_1_i<<<num_blocks, threads_per_block>>>(input, output, N, carries);

    // Second step: Compute offset for each thread group (exclusive scan for each thread group)
    scan_kernel_2<<<1, num_blocks>>>(carries);

    // Third step: Offset each thread group accordingly
    scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);

    cudaFree(carries);
}

// compute averages
float med(std::vector<float> log_vec)
{
    return log_vec[log_vec.size() / 2];
}

int main()
{

    int N=100;
    for (;N<100000001;N*=10)
    {
        //
        // Allocate host arrays for reference
        //
        double *x = (double *)malloc(sizeof(double) * N);
        double *y = (double *)malloc(sizeof(double) * N);
        double *z = (double *)malloc(sizeof(double) * N);
        double *z_i_b = (double *)malloc(sizeof(double) * N);
        double *z_i_c = (double *)malloc(sizeof(double) * N);
        std::fill(x, x + N, 1);

        // reference calculation:
        y[0] = 0;
        for (std::size_t i = 1; i < N; ++i)
            y[i] = y[i - 1] + x[i - 1];

        //
        // Allocate CUDA-arrays
        //
        double *cuda_x, *cuda_y, *cuda_y_c;
        cudaMalloc(&cuda_x, sizeof(double) * N);
        cudaMalloc(&cuda_y, sizeof(double) * N);
        cudaMalloc(&cuda_y_c, sizeof(double) * N);
        cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);

        // save data struc
        std::vector<float> log_ex;
        std::vector<float> log_ib;
        std::vector<float> log_ic;

        // initiate timer
        Timer timer;

        for (int j = 0; j < 11; j++)
        {

            // Perform the exclusive scan and obtain results
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            exclusive_scan(cuda_x, cuda_y, N);
            log_ex.push_back(timer.get());
            CUDA_ERRCHK(cudaDeviceSynchronize());

            cudaMemcpy(z, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

            // b) inclusive scan fully using exclusive scan
            std::fill(z_i_b, z_i_b + N, 0);

            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            for (int i = 0; i < N - 1; i++)
            {
                z_i_b[i] = z[i + 1];
            }
            z_i_b[N - 1] = z_i_b[N - 2] + x[N - 1];
            log_ib.push_back(timer.get() + log_ex.back());
            CUDA_ERRCHK(cudaDeviceSynchronize());

            // c) inclusive

            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            inclusive_scan(cuda_x, cuda_y_c, N);
            log_ic.push_back(timer.get());
            CUDA_ERRCHK(cudaDeviceSynchronize());
            cudaMemcpy(z_i_c, cuda_y_c, sizeof(double) * N, cudaMemcpyDeviceToHost);
        }

        // define medians
        float log_ex_av = med(log_ex);
        float log_ib_av = med(log_ib);
        float log_ic_av = med(log_ic);

        // output
        std::cout << N << " " << 1e3 * log_ex_av << " " << 1e3 * log_ib_av << " " << 1e3 * log_ic_av << std::endl; // milli seconds
    

        //
        // Print first few entries for reference
        //
        /*
        std::cout << "CPU y: ";
        for (int i = 0; i < 10; ++i)
            std::cout << y[i] << " ";
        std::cout << " ... ";
        for (int i = N - 10; i < N; ++i)
            std::cout << y[i] << " ";
        std::cout << std::endl;

        std::cout << "GPU y: ";
        for (int i = 0; i < 10; ++i)
            std::cout << z[i] << " ";
        std::cout << " ... ";
        for (int i = N - 10; i < N; ++i)
            std::cout << z[i] << " ";
        std::cout << std::endl;

        // b) print inclusive scan
        std::cout << "GPU y_i_b: ";
        for (int i = 0; i < 10; ++i)
            std::cout << z_i_b[i] << " ";
        std::cout << " ... ";
        for (int i = N - 10; i < N; ++i)
            std::cout << z_i_b[i] << " ";
        std::cout << std::endl;

        // c) print inclusive scan
        std::cout << "GPU y_i_c: ";
        for (int i = 0; i < 10; ++i)
            std::cout << z_i_c[i] << " ";
        std::cout << " ... ";
        for (int i = N - 10; i < N; ++i)
            std::cout << z_i_c[i] << " ";
        std::cout << std::endl;
        */

        //
        // Clean up:
        //
        free(x);
        free(y);
        free(z);
        cudaFree(cuda_x);
        cudaFree(cuda_y);
        cudaFree(cuda_y_c);
    }
    return EXIT_SUCCESS;
}
