#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <stdio.h>

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define WARP_SIZE 32

#define NUM_STAGES 2

__global__ void mmaOBTSKernelSparse_24sparse(
    half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, char *metadata,
    half *sparseMatrixA, half *B, half *C, size_t M, size_t N, size_t K,
    size_t nonzeroBlocks, int *blockInfo, int *relativeBlockIndexMapping) {
  const size_t K_tiles = div_ceil(K, MMA_K);
  // DEBUG_PRINT_THREAD(0, "(GPU) bcsrRowPtrA[1]: %d\n", bcsrRowPtrA[1]);

  // DEBUG_PRINT_THREAD(0, "(GPU) bcsrRowPtrA[0]: %d\n", bcsrRowPtrA[0]);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K - 1) / (MMA_K);

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  size_t start = bcsrRowPtrA[blockRow];
  size_t end = bcsrRowPtrA[blockRow + 1];

  __shared__ half A_smem[NUM_STAGES][MMA_M][MMA_K];
  __shared__ half B_smem[NUM_STAGES][MMA_N][MMA_K];
  __shared__ half C_smem[MMA_M][MMA_N];

  uint32_t sparsityStage0 = 0;
  uint32_t sparsityStage1 = 0;

  __shared__ half A_smem_sparse[NUM_STAGES][MMA_M][MMA_K / 2];
  __shared__ half B_smem_sparse[NUM_STAGES][MMA_N][MMA_K];
  __shared__ char Meta_smem_sparse[NUM_STAGES][MMA_M][MMA_K / 8];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][4];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  auto loadStages = [&] __device__(size_t stage_ptr, int stage) {
    if (stage_ptr < end) {

      size_t i = bcsrColIdxA[stage_ptr] / MMA_K;
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      int sparsityInfo = blockInfo[blockIndex];

      if (stage == 0) {
        sparsityStage0 = sparsityInfo;
      }

      if (stage == 1) {
        sparsityStage1 = sparsityInfo;
      }

      cuda::memcpy_async(
          ((int4 *)(&A_smem_sparse[stage][lane_id / 2][0]) + lane_id % 2),
          (((int4 *)(sparseMatrixA)) + relativeIndex * MMA_M * (MMA_K / 16) +
           lane_id),
          sizeof(int4), pipe);
      cuda::memcpy_async(
          ((half *)(Meta_smem_sparse[stage][lane_id / 2]) + (lane_id % 2)),
          ((half *)metadata + (relativeIndex * MMA_M * (MMA_K / 16) + lane_id)),
          sizeof(half), pipe);

      cuda::memcpy_async(
          ((int4 *)(&B_smem_sparse[stage][lane_id / 4][0]) + lane_id % 4),
          ((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 4) * K]) +
           lane_id % 4),
          sizeof(int4), pipe);
    }
  };

  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    loadStages(ptr, stage);

    pipe.producer_commit();
  }

  uint32_t RC[2] = {0, 0};

  int stage = 0;
  // int counter = 0;
  // DEBUG_PRINT_THREAD(0, "start_pointer: %d\n", bcsrRowPtrA[blockRow]);
  // DEBUG_PRINT_THREAD(0, "end_pointer: %d\n", bcsrRowPtrA[blockRow + 1]);

#pragma unroll
  for (size_t ptr = start; ptr < end; ptr++) {
    // counter++;
    // DEBUG_PRINT_THREAD(0, "counter: %d\n", counter);
    // DEBUG_PRINT_THREAD(0, "bcsrColIdxA[ptr]: %d\n", bcsrColIdxA[ptr]);

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
    size_t i = bcsrColIdxA[ptr] / MMA_K;
    // skip empty block
    size_t blockIndex = blockRow * colRegions + i;

    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    size_t A_size = MMA_M * MMA_K * sizeof(half);
    size_t B_size = MMA_N * MMA_K * sizeof(half);

    // int sparsityInfo = blockInfo[blockIndex];
    // int sparsityInfo = 1;
    // int sparsityInfo = sparsity1;
    int sparsityInfo = stage == 1 ? sparsityStage1 : sparsityStage0;
    // __syncthreads();

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
        &A_smem_sparse[stage][lane_id % 16][(lane_id / 16) * (MMA_K / 2 / 2)]);
    LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                A_smem_lane_addr);

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
        &B_smem_sparse[stage][lane_id % 8][((lane_id / 8) % 2) * (MMA_K / 2)]);
    LDMATRIX_X4(RB[stage][0], RB[stage][1], RB[stage][2], RB[stage][3],
                B_smem_lane_addr);

    char metadata_local[4];

    metadata_local[0] =
        (char)((Meta_smem_sparse[stage][lane_id / 4][0 + 2 * (lane_id % 2)]));
    metadata_local[1] =
        (char)((Meta_smem_sparse[stage][lane_id / 4][1 + 2 * (lane_id % 2)]));
    metadata_local[2] = (char)((
        Meta_smem_sparse[stage][(lane_id / 4) + 8][0 + 2 * (lane_id % 2)]));
    metadata_local[3] = (char)((
        Meta_smem_sparse[stage][(lane_id / 4) + 8][1 + 2 * (lane_id % 2)]));

    uint32_t meta_value;
    memcpy(&meta_value, metadata_local, sizeof(uint32_t));

    // HMMA16816_SPARSE(RC[0], RC[1], RA[stage][0], RA[stage][1],
    // RB[stage][0],
    //                  RB[stage][1], RC[0], RC[1], meta_value, 0x0);

    HMMA16832_SPARSE(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
                     RA[stage][3], RB[stage][0], RB[stage][1], RB[stage][2],
                     RB[stage][3], RC[0], RC[1], meta_value, 0x0);

    // __syncthreads();

    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    loadStages(stage_ptr, stage);

    pipe.producer_commit();

    stage = (stage + 1) % NUM_STAGES;
  }

  *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4 *)(&C_smem[lane_id][0]));
  }
}

__global__ void mmaOBTSKernelDense_dense(half *bcsrValuesA, int *bcsrRowPtrA,
                                         int *bcsrColIdxA, char *metadata,
                                         half *sparseMatrixA, half *B, half *C,
                                         size_t M, size_t N, size_t K,
                                         size_t nonzeroBlocks, int *blockInfo,
                                         int *relativeBlockIndexMapping) {
  const size_t K_tiles = div_ceil(K, MMA_K);
  // DEBUG_PRINT_THREAD(0, "(GPU) bcsrRowPtrA[1]: %d\n", bcsrRowPtrA[1]);

  // DEBUG_PRINT_THREAD(0, "(GPU) bcsrRowPtrA[0]: %d\n", bcsrRowPtrA[0]);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K - 1) / (MMA_K);

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  size_t start = bcsrRowPtrA[blockRow];
  size_t end = bcsrRowPtrA[blockRow + 1];

  __shared__ half A_smem[NUM_STAGES][MMA_M][MMA_K];
  __shared__ half B_smem[NUM_STAGES][MMA_N][MMA_K];
  __shared__ half C_smem[MMA_M][MMA_N];

  uint32_t sparsityStage0 = 0;
  uint32_t sparsityStage1 = 0;

  __shared__ half A_smem_sparse[NUM_STAGES][MMA_M][MMA_K / 2];
  __shared__ half B_smem_sparse[NUM_STAGES][MMA_N][MMA_K];
  __shared__ char Meta_smem_sparse[NUM_STAGES][MMA_M][MMA_K / 8];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][4];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  auto loadStages = [&] __device__(size_t stage_ptr, int stage) {
    if (stage_ptr < end) {

      size_t i = bcsrColIdxA[stage_ptr] / MMA_K;
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      int sparsityInfo = blockInfo[blockIndex];

      if (stage == 0) {
        sparsityStage0 = sparsityInfo;
      }

      if (stage == 1) {
        sparsityStage1 = sparsityInfo;
      }

      cuda::memcpy_async(
          ((long4 *)(&A_smem[stage][lane_id / 2][0]) + lane_id % 2),
          (((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                   (lane_id / 2) * MMA_K]) +
            lane_id % 2)),
          sizeof(long4), pipe);

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        cuda::memcpy_async(
            ((long4 *)(&B_smem[stage][lane_id / 2][0]) + lane_id % 2),
            ((long4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
             lane_id % 2),
            sizeof(long4), pipe);
      }
    }
  };

  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    loadStages(ptr, stage);

    pipe.producer_commit();
  }

  uint32_t RC[2] = {0, 0};

  int stage = 0;
  // int counter = 0;
  // DEBUG_PRINT_THREAD(0, "start_pointer: %d\n", bcsrRowPtrA[blockRow]);
  // DEBUG_PRINT_THREAD(0, "end_pointer: %d\n", bcsrRowPtrA[blockRow + 1]);

#pragma unroll
  for (size_t ptr = start; ptr < end; ptr++) {
    // counter++;
    // DEBUG_PRINT_THREAD(0, "counter: %d\n", counter);
    // DEBUG_PRINT_THREAD(0, "bcsrColIdxA[ptr]: %d\n", bcsrColIdxA[ptr]);

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
    size_t i = bcsrColIdxA[ptr] / MMA_K;
    // skip empty block
    size_t blockIndex = blockRow * colRegions + i;

    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    size_t A_size = MMA_M * MMA_K * sizeof(half);
    size_t B_size = MMA_N * MMA_K * sizeof(half);

    // int sparsityInfo = blockInfo[blockIndex];
    // int sparsityInfo = 1;
    // int sparsityInfo = sparsity1;
    int sparsityInfo = stage == 1 ? sparsityStage1 : sparsityStage0;
    // __syncthreads();

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
        &A_smem[stage][lane_id % 16][(lane_id / 16) * 8]);
    LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                A_smem_lane_addr);

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
        &B_smem[stage][lane_id % 8][((lane_id / 8) % 2) * 8]);
    LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

    HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
              RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

    A_smem_lane_addr = __cvta_generic_to_shared(
        &A_smem[stage][lane_id % 16][(lane_id / 16) * 8 + 16]);
    LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                A_smem_lane_addr);

    B_smem_lane_addr = __cvta_generic_to_shared(
        &B_smem[stage][lane_id % 8][((lane_id / 8) % 2) * 8 + 16]);
    LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

    HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
              RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

    // __syncthreads();

    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    loadStages(stage_ptr, stage);

    pipe.producer_commit();

    stage = (stage + 1) % NUM_STAGES;
  }

  *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4 *)(&C_smem[lane_id][0]));
  }
}

// Kernel to combine results
__global__ void combineKernel(half *C_sparse, half *C_dense, half *C_final,
                              size_t totalElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalElements) {
    C_final[idx] = C_sparse[idx] + C_dense[idx];
  }
}

void mmaOBTSKernel_large_separate(
    half *bcsrValuesA_sparse, int *bcsrRowPtrA_sparse, int *bcsrColIdxA_sparse,
    half *bcsrValuesA_dense, int *bcsrRowPtrA_dense, int *bcsrColIdxA_dense,
    char *metadata, half *sparseMatrixA, half *B, half *C, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

  // Allocate temporary device buffers
  half *C_sparse = nullptr;
  half *C_dense = nullptr;
  const size_t matrix_size = M * N * sizeof(half);

  // Allocate temporary buffers
  cudaMalloc(&C_sparse, matrix_size);
  cudaMalloc(&C_dense, matrix_size);

  // Zero-initialize the buffers
  cudaMemset(C_sparse, 0, matrix_size);
  cudaMemset(C_dense, 0, matrix_size);

  mmaOBTSKernelSparse_24sparse<<<grid, block>>>(
      bcsrValuesA_sparse, bcsrRowPtrA_sparse, bcsrColIdxA_sparse, metadata,
      sparseMatrixA, B, C_sparse, M, N, K, nonzeroBlocks, blockInfo,
      relativeBlockIndexMapping);
  mmaOBTSKernelDense_dense<<<grid, block>>>(
      bcsrValuesA_dense, bcsrRowPtrA_dense, bcsrColIdxA_dense, metadata,
      sparseMatrixA, B, C_dense, M, N, K, nonzeroBlocks, blockInfo,
      relativeBlockIndexMapping);

  // Combine the results
  const int threadsPerBlock = 256;
  const int numBlocks = div_ceil(M * N, threadsPerBlock);
  combineKernel<<<numBlocks, threadsPerBlock>>>(C_sparse, C_dense, C, M * N);

  // Free temporary buffers
  cudaFree(C_sparse);
  cudaFree(C_dense);
}