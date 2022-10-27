#include <stdio.h>
#include "timer.hpp"


__global__ 
void sum_every_kth_entry(int n, int k, double *x, double *y) 
{
	unsigned int total_threads = blockDim.x * gridDim.x;
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		
	for (unsigned int i = thread_id; i < n; i += total_threads)
	{
		if (k > 0) {
			if ((i+1)%k == 0) {
				x[i] += y[i];
			}
		} else if (k == 0) {
			x[i] += y[i];
		}
	}
}

__global__ 
void skip_first_k_elements(int n, int k, double *x, double *y) 
{
	unsigned int total_threads = blockDim.x * gridDim.x;
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
		
	for (unsigned int i = thread_id; i < n; i += total_threads)
	{
		if ((i+1) > k) {
			x[i] += y[i];
		}
	}
}

double effBW(int mode, int N, int k, double time) {
	/* compute effective bandwidth for vector addition with
	  (a) Stride k ... mode == 0
	  (b) Offset k ... mode == 1
	  factor of 3 comes from number of read (2) and write (1) OPs
	*/
	size_t bytes = sizeof(double);
	double effBandwidth;
	
	if (mode == 0) {
		if (k > 0) {
			effBandwidth = (3 * (N/k) * bytes) / (1e9 * time);
		} else if (k == 0) {
			effBandwidth = (3 * N * bytes) / (1e9 * time);
		}
	} else if (mode == 1) {
		effBandwidth = (3 * (N-k) * bytes) / (1e9 * time);
	}
	
	return effBandwidth; 
}


int main(void)
{
  /* select kernel to launch (a) or (b) */
  int mode = 1;

  int N = 1e8, k = 0;
  size_t bytes = sizeof(double);
  double *x, *y, *d_x, *d_y, time;
  Timer timer;

  // Allocate host memory and initialize
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));
  
  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 1.0;
  }

  // Allocate device memory and copy host data over
  cudaMalloc(&d_x, N*sizeof(double)); 
  cudaMalloc(&d_y, N*sizeof(double));

  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  printf("k,time,effBW\n");
  while (k < 64) { 
	  // wait for previous operations to finish, then start timings
	  cudaDeviceSynchronize();
	  timer.reset();
	  for (int i = 0; i < 10; i++) {

		  if (mode == 0) {
			  // sum only every k-th entry of two vectors
			  sum_every_kth_entry<<<256, 256>>>(N, k, d_x, d_y);
		  } else if (mode == 1) {
			  // sum two vectors skipping first k entries
			  skip_first_k_elements<<<256, 256>>>(N, k, d_x, d_y);
		  } 

		  // wait for kernel to finish, then print elapsed time
		  cudaDeviceSynchronize();
	  
	  }
	  time = 0.1*timer.get();
	  printf("%d,%g,%g\n", k, time, effBW(mode, N, k, time));
	  k++;
  }

  // copy data back (implicit synchronization point)
  cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

  // tidy up host and device memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  cudaDeviceReset();  // for leak check to work for CUDA buffers
  return EXIT_SUCCESS;
}

