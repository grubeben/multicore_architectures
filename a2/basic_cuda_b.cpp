

#include <stdio.h>
#include "timer.hpp"

// get thread_id
__global__ void saxpy(int n, double a, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv)
{
    int N = 1000000;
    if (argc > 1)
    {
        std::cout << "You have entered" << argc << "arguments:"
                  << "\n";
        for (int i = 1; i = argc; i++)
        {
            std::cout << atoi(argv[i]) << std::endl;
        }
        int N = atoi(argv[2]);
    }

    double *x, *y, *d_x, *d_y;
    Timer timer;

    // Allocate host memory and initialize
    x = (double *)malloc(N * sizeof(double));
    y = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = N - i;
    }

    // Allocate device memory and copy host data over
      cudaMalloc(&d_x, N*sizeof(double); 
      cudaMalloc(&d_y, N*sizeof(double);
     
      cudaMemcpy(d_x, x, N*sizeof(double, cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N*sizeof(double, cudaMemcpyHostToDevice);
     
      // wait for previous operations to finish, then start timings
      cudaDeviceSynchronize();
      timer.reset();
     
     
      // ensure we have enough threads

      int threads_per_block=256;
      int num_threads=256;
      int num_blocks=(N+num_threads-1)/num_threads;

      // Perform SAXPY on 1M elements
      saxpy<<<num_blocks, threads_per_block>>>(N, 2.0, d_x, d_y);
     
      // wait for kernel to finish, then print elapsed time
      cudaDeviceSynchronize();
      printf("Elapsed: %g\n", timer.get());
     
      // copy data back (implicit synchronization point)
      cudaMemcpy(y, d_y, N*sizeof(double, cudaMemcpyDeviceToHost);
     
      // Numerical error check:
      double maxError = 0.0f;
      for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-4.0f));
      printf("Max error: %f\n", maxError);
     
      // tidy up host and device memory
      cudaFree(d_x);
      cudaFree(d_y);
      free(x);
      free(y);
     
      return EXIT_SUCCESS;
}
