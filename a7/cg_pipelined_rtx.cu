#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}

// features 3 vec_adds and 1 dp
__global__ void kernel1(int N, double *Ap, double *x, double *r, double *p, 
                        double alpha, double beta, double *rr)
{
    __shared__ double shared_mem[512];
    double dot = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        // vecadds
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
        p[i] = r[i] + beta * p[i]; // is r[i] guaranteed to be be r[i] computed?
        // dotp
        dot += r[i] * r[i];
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
        atomicAdd(rr, shared_mem[0]);
}

// features 1 matvec and 2 dps
__global__ void kernel2(int N, int *A_csr_rowoffsets, int *A_csr_colindices, 
                        double *A_csr_values, double *p, double *Ap, double *ApAp,
                        double *pAp)
{
    __shared__ double shared_mem1[512];
    __shared__ double shared_mem2[512];
    double dot1 = 0;
    double dot2 = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        double sum = 0;
        // matvec
        for (int k = A_csr_rowoffsets[i]; k < A_csr_rowoffsets[i + 1]; k++)
        {
            sum += A_csr_values[k] * p[A_csr_colindices[k]];
        }
        Ap[i] = sum;

        // dotp
        dot1 += Ap[i] * Ap[i];
        dot2 += p[i] * Ap[i];
    }

    shared_mem1[threadIdx.x] = dot1;
    shared_mem2[threadIdx.x] = dot2;
    for (int k = blockDim.x / 2; k > 0; k /= 2)
    {
        __syncthreads();
        if (threadIdx.x < k)
        {
            shared_mem1[threadIdx.x] += shared_mem1[threadIdx.x + k];
            shared_mem2[threadIdx.x] += shared_mem2[threadIdx.x + k];
        }
    }
    if (threadIdx.x == 0)
        atomicAdd(ApAp, shared_mem1[0]);
        atomicAdd(pAp, shared_mem2[0]);
}

/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with CUDA. Modify as you see fit.
 */
void conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
    // initialize timer
    Timer timer;

    // clear solution vector (it may contain garbage values):
    std::fill(solution, solution + N, 0);

    // initialize work vectors:
    double alpha, beta, pAp, rr;
    double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_rr, *cuda_ApAp, *cuda_pAp;
    cudaMalloc(&cuda_p, sizeof(double) * N);
    cudaMalloc(&cuda_r, sizeof(double) * N);
    cudaMalloc(&cuda_Ap, sizeof(double) * N);
    cudaMalloc(&cuda_solution, sizeof(double) * N);
    cudaMalloc(&cuda_rr, sizeof(double));
    cudaMalloc(&cuda_ApAp, sizeof(double));
    cudaMalloc(&cuda_pAp, sizeof(double));

    cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

    //compute rr0
    cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_rr);
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);

    //compute alpha, beta, Ap once prior to interating ==> this doesnt satisfy "2 kernels per iteration"
    kernel2<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap, cuda_ApAp, cuda_pAp);
    cudaMemcpy(&beta, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    alpha = rr/pAp;
    beta *=alpha*alpha;
    beta=beta-rr;
    beta/=rr;
    
    printf("\nstarting alpha, beta: %g, %g\n", alpha, beta);
    printf("\nstarting rr: %g\n", rr);

    double initial_residual_squared = rr;
    int iters = 0;
    cudaDeviceSynchronize();
    timer.reset();
    while (1)
    {
        // lines 2-4
        kernel1<<<512, 512>>>(N, cuda_Ap, cuda_solution, cuda_r, cuda_p, alpha, beta, cuda_rr);

        cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);
        //for (int l=0; l<N; l++)printf("%g\n", solution[l]);
        printf("\n");

        // lines 5,6
        kernel2<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap, 
                            cuda_ApAp, cuda_pAp);
        //copy to host
        cudaMemcpy(&beta, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
        //compute alpha, beta
        alpha = rr/pAp;
        beta *=alpha*alpha;
        beta-=rr;
        beta/=rr;

        printf("\n rr: %g\n", rr);
        // line 10:
        if (std::sqrt(rr / initial_residual_squared) < 1e-6)
        {
            break;
        }

        if (iters > 10)
            break; // solver didn't converge
        ++iters;
    }
    cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

    if (iters > 10)
        std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
                  << std::endl;
    else
        std::cout << "Conjugate Gradient converged in " << iters << " iterations."
                  << std::endl;

    cudaFree(cuda_p);
    cudaFree(cuda_r);
    cudaFree(cuda_Ap);
    cudaFree(cuda_solution);
    cudaFree(cuda_rr);
    cudaFree(cuda_ApAp);
    cudaFree(cuda_pAp);
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction)
{

    int N = points_per_direction *
            points_per_direction; // number of unknows to solve for

    std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

    //
    // Allocate CSR arrays.
    //
    // Note: Usually one does not know the number of nonzeros in the system matrix
    // a-priori.
    //       For this exercise, however, we know that there are at most 5 nonzeros
    //       per row in the system matrix, so we can allocate accordingly.
    //
    int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
    int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
    double *csr_values = (double *)malloc(sizeof(double) * 5 * N);

    int *cuda_csr_rowoffsets, *cuda_csr_colindices;
    double *cuda_csr_values;
    //
    // fill CSR matrix with values
    //
    generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
                         csr_values);

    //
    // Allocate solution vector and right hand side:
    //
    double *solution = (double *)malloc(sizeof(double) * N);
    double *rhs = (double *)malloc(sizeof(double) * N);
    std::fill(rhs, rhs + N, 1);

    //
    // Allocate CUDA-arrays //
    //
    cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
    cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
    cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
    cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);

    //
    // Call Conjugate Gradient implementation with GPU arrays
    //
    conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);

    //
    // Check for convergence:
    //
    double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
    std::cout << "Relative residual norm: " << residual_norm
              << " (should be smaller than 1e-6)" << std::endl;

    cudaFree(cuda_csr_rowoffsets);
    cudaFree(cuda_csr_colindices);
    cudaFree(cuda_csr_values);
    free(solution);
    free(rhs);
    free(csr_rowoffsets);
    free(csr_colindices);
    free(csr_values);
}

int main()
{

    solve_system(3); // solves a system with 100*100 unknowns

    return EXIT_SUCCESS;
}
