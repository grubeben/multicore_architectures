
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"

/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
__global__ void cuda_csr_matvec_product(size_t N,
                                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N; row += gridDim.x * blockDim.x)
  {
    double val = 0;
    for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row + 1]; ++jj)
    {
      val += csr_values[jj] * x[csr_colindices[jj]];
    }
    y[row] = val;
  }
}

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
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
  {
    atomicAdd(result, shared_mem[0]);
  }
}

__global__ void vecAdd7(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] += alpha * y[i];
  }
}

__global__ void vecAdd8(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] -= alpha * y[i];
  }
}

__global__ void vecAdd12(int N, double *x, double *y, double beta)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    x[i] = y[i] + beta * x[i];
  }
}

/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use with CUDA.
 *  Modify as you see fit.
 */
void conjugate_gradient(size_t N, // number of unknows
                        int *csr_rowoffsets, int *csr_rowoffsets_d, int *csr_colindices, int *csr_colindices_d, double *csr_values, double *csr_values_d,
                        double *rhs, double *rhs_d,
                        double *solution, double *solution_d)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{

  const int nBlocks = 256;
  const int nThreads = 256;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double *p = (double *)malloc(sizeof(double) * N);
  double *r = (double *)malloc(sizeof(double) * N);
  double *Ap = (double *)malloc(sizeof(double) * N);
  double alpha = 0;
  double beta = 0;
  double pAp = 0;
  double rr = 0;
  double rr_old;

  // initialize work vectors for GPU
  double *p_d, *r_d, *cuda_rr, *Ap_d, *cuda_pAp, *cuda_alpha;
  cudaMalloc(&p_d, sizeof(double) * N);
  cudaMalloc(&r_d, sizeof(double) * N);
  cudaMalloc(&Ap_d, sizeof(double) * N);
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));
  cudaMalloc(&cuda_alpha, sizeof(double));

  // line 2: initialize r and p:
  std::copy(rhs, rhs + N, p);
  std::copy(rhs, rhs + N, r);

  // copy data to device
  cudaMemcpy(solution_d, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(p_d, p, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(r_d, r, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_pAp, &pAp, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Ap_d, Ap, sizeof(double) * N, cudaMemcpyHostToDevice);

  int iters = 0;
  while (1)
  {
    // line 4: A*p:
    cuda_csr_matvec_product<<<nBlocks, nThreads>>>(N, csr_rowoffsets_d, csr_colindices_d, csr_values_d, p_d, Ap_d);

    rr = 0;
    cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
    pAp = 0;
    cudaMemcpy(cuda_pAp, &pAp, sizeof(double), cudaMemcpyHostToDevice);

    // line 5:
    cuda_dot_product<<<nBlocks, nThreads>>>(N, p_d, Ap_d, cuda_pAp);
    // line 6:
    cuda_dot_product<<<nBlocks, nThreads>>>(N, r_d, r_d, cuda_rr);

    cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

    rr_old = rr;

    // line 6:
    alpha = rr / pAp;

    // line 7:
    vecAdd7<<<nBlocks, nThreads>>>(N, solution_d, p_d, alpha);

    // line 8:
    vecAdd8<<<nBlocks, nThreads>>>(N, r_d, Ap_d, alpha);
    cudaMemcpy(r, r_d, sizeof(double) * N, cudaMemcpyDeviceToHost);

    rr = 0;
    cudaMemcpy(cuda_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
    // line 9:
    cuda_dot_product<<<nBlocks, nThreads>>>(N, r_d, r_d, cuda_rr); // r_d is already updated at this point (from last kernel)

    // line 10:
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

    if (rr < 1e-7)
    {
      cudaMemcpy(solution, solution_d, N * sizeof(double), cudaMemcpyDeviceToHost);
      break;
    }

    // line 11:
    beta = rr / rr_old;

    // line 12:
    vecAdd12<<<nBlocks, nThreads>>>(N, p_d, r_d, beta);
    cudaMemcpy(p, p_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations" << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations." << std::endl;

  free(p);
  free(r);
  free(Ap);
  cudaFree(p_d);
  cudaFree(r_d);
  cudaFree(Ap_d);
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction)
{

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);
  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  // Allocate GPU memory
  int *csr_rowoffsets_d;
  int *csr_colindices_d;
  double *csr_values_d;
  double *solution_d;
  double *rhs_d;
  cudaMalloc(&csr_rowoffsets_d, sizeof(double) * (N + 1));
  cudaMalloc(&csr_colindices_d, sizeof(double) * 5 * N);
  cudaMalloc(&csr_values_d, sizeof(double) * 5 * N);
  cudaMalloc(&solution_d, sizeof(double) * N);
  cudaMalloc(&rhs_d, sizeof(double) * N);

  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  // copy host arrays to GPU
  cudaMemcpy(csr_rowoffsets_d, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_colindices_d, csr_colindices, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
  cudaMemcpy(csr_values_d, csr_values, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
  cudaMemcpy(solution_d, solution, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_d, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
  //
  conjugate_gradient(N, csr_rowoffsets, csr_rowoffsets_d, csr_colindices, csr_colindices_d, csr_values, csr_values_d, rhs, rhs_d, solution, solution_d);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;

  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
  cudaFree(solution_d);
  cudaFree(rhs_d);
  cudaFree(csr_rowoffsets_d);
  cudaFree(csr_colindices_d);
  cudaFree(csr_values_d);
}

int main()
{

  Timer timer;
  int N_vals[6] = {100};
  for (int i = 0; i < 1; i++)
  {

    int N = N_vals[i];
    timer.reset();

    solve_system(N); // solves a system with 100*100 unknowns

    float t = timer.get();

    printf("\nTIME: %f\n", t);
  }

  return EXIT_SUCCESS;
}