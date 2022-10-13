

#include <stdio.h>
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <numeric>

// device operation
__global__ void saxpy(int n, double a, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}


//allocate device memory
void alloc_dev_mem(int N, double* d_x, double* d_y){
    cudaMalloc(&d_x, N*sizeof(double)); 
    cudaMalloc(&d_y, N*sizeof(double));
}


//free device memory



int main(int argc, char **argv)
{
    int N = 1000000;
    if (argc > 1)
    {
        std::cout << "You have entered" << argc << "arguments:"<< "\n";
        for (int i = 0; i = argc; i++)
        {
            std::cout << atoi(argv[i]) << std::endl;
        }
        int N = atoi(argv[1]);
    }

    //initialize
    double *x, *y, *d_x, *d_y;
    Timer timer;

    // Allocate host memory and initialize
    x = (double *)malloc(N * sizeof(double));
    y = (double *)malloc(N * sizeof(double));

    //set values on host side
    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = N - i;
    }

    std::vector <float> log_al;
    std::vector <float> log_fr;

    for (int i= 0; i<10;i++){

        //start timer
        cudaDeviceSynchronize();
        timer.reset();

        // Allocate device memory and copy host data over
        cudaMalloc(&d_x, N*sizeof(double)); 
        cudaMalloc(&d_y, N*sizeof(double));

        // wait for kernel to finish, then log elapsed time
        cudaDeviceSynchronize();
        float elapsed_time_al = timer.get();
        log_al.push_back(elapsed_time_al);

        //printf("Elapsed while allocating: %g\n", elapsed_time_al);
        
        cudaDeviceSynchronize();
        timer.reset();

        cudaFree(d_x);
        cudaFree(d_y);

        // wait for kernel to finish, then log elapsed time
        cudaDeviceSynchronize();
        float elapsed_time_fr = timer.get();
        log_fr.push_back(elapsed_time_al);
        //printf("Elapsed while freeing: %g\n", elapsed_time_fr);
    }
    float log_al_av = std::accumulate(log_al.begin(),log_al.end(),0.0/log_al.size());
    float log_fr_av= std::accumulate(log_fr.begin(),log_fr.end(),0.0/log_fr.size());

    std::cout<<"time to allocate:"<<log_al_av<<"\n"<<"time to free:" <<log_al_av<<std::endl;
    
    free(x);
    free(y);
    
    return EXIT_SUCCESS;
}
