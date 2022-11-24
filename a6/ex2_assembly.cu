#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "cuda_errchk.hpp"

__global__ void count_nnz(int *nn_counts, int N, int M)
// each row describes one node, hence simply by the position of the row in the matrix,
// we can deduce how many entries are expected to be populated
{
    for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N * M; row += gridDim.x * blockDim.x)
    {
        int nnz_for_this_node = 1;
        int i = row / N;
        int j = row % N;

        if (i > 0)
            nnz_for_this_node += 1;
        if (j > 0)
            nnz_for_this_node += 1;
        if (i < N - 1)
            nnz_for_this_node += 1;
        if (j < M - 1)
            nnz_for_this_node += 1;

        nn_counts[row] = nnz_for_this_node;
    }
}

__global__ void populate_matrix(int *row_offsets,int *values,int *col_indices,int N,int M)
{
    for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) 
    {
        int i = row / N;
        int j = row % N;
        int this_row_offset = row_offsets[row];

        // diagonal entry
        col_indices[this_row_offset] = i * N + j;
        values[this_row_offset] = 4;
        this_row_offset += 1;

        // upper neighbor
        if (i > 0) 
        { 
        col_indices[this_row_offset] = (i-1)* N+j;
        values[this_row_offset] = -1;
        this_row_offset += 1;
        }

        // left neighbor
        if (j > 0) 
        { 
        col_indices[this_row_offset] = i* N +(j-1);
        values[this_row_offset] = -1;
        this_row_offset += 1;
        }

        // lower neighbor
        if (i < N-1) 
        { 
        col_indices[this_row_offset] = (i+1)* N +j;
        values[this_row_offset] = -1;
        this_row_offset += 1;
        }

        // right neighbour
        if (j < M-1) 
        { 
        col_indices[this_row_offset] = i+ N *(j+1);
        values[this_row_offset] = -1;
        this_row_offset += 1;
        }
    }
}

__global__ void scan_kernel_1(int const *X, int *Y, int N, int *carries)
{
    __shared__ int shared_buffer[256];
    int my_value;

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

// exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
    __shared__ int shared_buffer[256];

    // load data:
    int my_carry = carries[threadIdx.x];

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

__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
    unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
    unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
    unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);

    __shared__ int shared_offset;

    if (threadIdx.x == 0)
        shared_offset = carries[blockIdx.x];

    __syncthreads();

    // add offset to each element in the block:
    for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
        if (i < N)
            Y[i] += shared_offset;
}

void exclusive_scan(int const *input,
                    int *output, int N)
{
    int num_blocks = 256;
    int threads_per_block = 256;

    int *carries;
    cudaMalloc(&carries, sizeof(int) * num_blocks);

    // First step: Scan within each thread group and write carries
    scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);

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

    int N = 3;
    for (; N < 11; N *= 10)
    {
        // Allocation sizes
        int n_values = 5*(N-2)*(N-2)+4*4*(N-2)+4*3; //5*N*N is definitely sufficient, can we go exact? yes
        // Allocate host arrays
        int *row_offsets = (int *)malloc(sizeof(int) * N * N); // N*M nodes ==> N*M rows
        int *nn_counts = (int *)malloc(sizeof(int) *N* N);
        int *values = (int *)malloc(sizeof(int) *n_values);
        int *col_indices = (int *)malloc(sizeof(int) *n_values); 

        //for reference CPU application
        int *row_offsets_cpu = (int *)malloc(sizeof(int) * N * N); // N*M nodes ==> N*M rows
        int *nn_counts_cpu = (int *)malloc(sizeof(int) *N* N);
        double *values_cpu = (double *)malloc(sizeof(double) *n_values);
        int *col_indices_cpu = (int *)malloc(sizeof(int) *n_values); 

        // Allocate CUDA-arrays
        int *cuda_row_offsets, *cuda_nn_counts, *cuda_values, *cuda_col_indices;
        cudaMalloc(&cuda_row_offsets, sizeof(int) *N* N);
        cudaMalloc(&cuda_nn_counts, sizeof(int) *N* N);
        cudaMalloc(&cuda_values, sizeof(int) *n_values);
        cudaMalloc(&cuda_col_indices, sizeof(int) *n_values);

        // save data struc
        std::vector<float> log_assembly;

        // initiate timer
        Timer timer;

        for (int j = 0; j < 11; j++)
        {
            // Matrix assembly
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            count_nnz<<<256, 256>>>(cuda_nn_counts, N, N); //a)
            exclusive_scan(cuda_nn_counts,cuda_row_offsets, N*N); //b)
            populate_matrix<<<256,256>>>(cuda_row_offsets,cuda_values,cuda_col_indices,N,N); //c)
            log_assembly.push_back(timer.get());
            CUDA_ERRCHK(cudaDeviceSynchronize());

            //reference CPU version
            generate_fdm_laplace(N, row_offsets_cpu, col_indices_cpu, values_cpu);
                
            //copy back to CPU
            cudaMemcpy(nn_counts, cuda_nn_counts, sizeof(int)*N * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(row_offsets, cuda_row_offsets, sizeof(int)*N * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(values, cuda_values, sizeof(int)*n_values, cudaMemcpyDeviceToHost);
            cudaMemcpy(col_indices, cuda_col_indices, sizeof(int)*n_values, cudaMemcpyDeviceToHost);
            
        }

        // define median
        float log_assembly_av = med(log_assembly);

        // output
        std::cout << N << " " << 1e3 * log_assembly_av << std::endl; // milli seconds



        
        // OUTPUT FOR VALIDATION (small N)
        std::cout << "nn_counts:\n";
        for (int i = 0; i < N*N; ++i)
            std::cout << nn_counts[i] << std::endl;
        std::cout << "\n";

        std::cout << "values:\n";
        for (int i = 0; i < n_values; ++i)
            std::cout << values[i]<< " "<<values_cpu[i] << std::endl;
        std::cout << "\n";

        std::cout << "col indeces:\n";
        for (int i = 0; i < n_values; ++i)
            std::cout << col_indices[i]<< " "<<col_indices_cpu[i]  << std::endl;
        std::cout << "\n";

        std::cout << "row offsets:\n";
        for (int i = 0; i < N*N; ++i)
            std::cout << row_offsets[i]<< " "<<row_offsets_cpu[i] << std::endl;
        

        

        // Clean up:
        free(row_offsets);
        free(row_offsets_cpu);
        free(values);
        free(values_cpu);
        free(nn_counts);
        free(col_indices);
        free(col_indices_cpu);
        cudaFree(cuda_row_offsets);
        cudaFree(cuda_nn_counts);
        cudaFree(cuda_values);
        cudaFree(cuda_col_indices);
    }
    return EXIT_SUCCESS;
}
