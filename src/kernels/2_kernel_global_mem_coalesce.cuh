#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * @brief SGEMM kernel using global memory with coalesced access pattern
 * 
 * This kernel implements matrix multiplication using global memory access.
 * The key optimization here is the thread mapping strategy that enables coalesced memory access
 * for matrix B, which significantly improves memory bandwidth utilization.
 * 
 * @tparam BLOCKSIZE The size of the thread block (must be a perfect square)
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in matrix B and C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param alpha Scalar multiplier for A*B
 * @param A Input matrix A (MxK)
 * @param B Input matrix B (KxN)
 * @param beta Scalar multiplier for C
 * @param C Input/output matrix C (MxN)
 */
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // Calculate the row and column indices in matrix C that this thread will compute
  // This mapping ensures that threads in the same warp access consecutive elements
  // in matrix B, enabling coalesced memory access
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // Boundary check to ensure we don't compute elements outside the matrix dimensions
  // This is necessary because the grid might be larger than the matrix size
  if (cRow < M && cCol < N) {
    // Accumulator for the dot product
    float tmp = 0.0;
    
    // Compute the dot product for one element of C
    // For each element C[cRow][cCol], we need to multiply and sum:
    // A[cRow][0] * B[0][cCol] + A[cRow][1] * B[1][cCol] + ... + A[cRow][K-1] * B[K-1][cCol]
    for (int i = 0; i < K; ++i) {
      // Access pattern:
      // - A: Strided access (not coalesced) as each thread in a warp accesses different rows
      // - B: Coalesced access as threads in the same warp access consecutive columns
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    
    // Store the result with scaling factors alpha and beta
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}