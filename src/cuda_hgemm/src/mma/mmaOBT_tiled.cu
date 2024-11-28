#pragma once

#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <stdio.h>

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCKSIZE 2

#define WARP_SIZE 32

#define NUM_STAGES 2

__global__ void mmaOBTKernelSparse_tiled(half *bcsrValuesA, int *bcsrRowPtrA,
                                         int *bcsrColIdxA, half *B, half *C,
                                         size_t M, size_t N, size_t K,
                                         size_t nonzeroBlocks, int *blockInfo,
                                         int *relativeBlockIndexMapping) {
  // mmaCBTKernel
  const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M * BLOCKSIZE;
  const size_t warp_col = blockIdx.x * MMA_N * BLOCKSIZE;

  size_t blockRow = blockIdx.y * BLOCKSIZE;
  size_t blockCol = blockIdx.x * BLOCKSIZE;

  // size_t colRegions = (K + MMA_K - 1) / (MMA_K);
  size_t colRegions = (K + MMA_K * BLOCKSIZE - 1) / (MMA_K * BLOCKSIZE);

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  int this_ptr = bcsrRowPtrA[blockRow];

  int next_ptr = bcsrRowPtrA[blockRow + 1];

  __shared__ half A_smem[NUM_STAGES][BLOCKSIZE][BLOCKSIZE][MMA_M][MMA_K];
  __shared__ half B_smem[NUM_STAGES][BLOCKSIZE][BLOCKSIZE][MMA_N][MMA_K];
  __shared__ half C_smem[BLOCKSIZE][BLOCKSIZE][MMA_M][MMA_N];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  // tileIdX should be in [0, BLOCKSIZE_X)
  const size_t tileIdX = (threadIdx.x / WARP_SIZE) % BLOCKSIZE;
  // tileIdY should be in [0, BLOCKSIZE_Y)
  const size_t tileIdY = threadIdx.x / (WARP_SIZE * BLOCKSIZE);

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][2];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = this_ptr + stage;
    // printf("ptr %d\n", ptr);
    // printf("next_ptr %d\n", next_ptr);
    if (ptr < next_ptr) {
      size_t i = bcsrColIdxA[ptr] / MMA_K;
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      // printf("before 1st memcpy_async");

      cuda::memcpy_async(
          ((int4 *)(&A_smem[stage][tileIdY][tileIdX][lane_id / 2][0]) +
           lane_id % 2),
          (((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * BLOCKSIZE * MMA_K *
                                      BLOCKSIZE +
                                  (tileIdY * MMA_K * MMA_M * BLOCKSIZE +
                                   tileIdX * MMA_M * MMA_K) +
                                  (lane_id / 2) * MMA_K]) +
            lane_id % 2)),
          sizeof(int4), pipe);
      // printf("after 1st memcpy_async");

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][tileIdY][tileIdX][lane_id / 2][0]) +
             lane_id % 2),
            ((int4 *)(&B[i * MMA_K * BLOCKSIZE + tileIdX * MMA_K +
                         (warp_col + tileIdY * WARP_SIZE + lane_id / 2) * K]) +
             lane_id % 2),
            sizeof(int4), pipe);
      }

      pipe.producer_commit();
    }
  }

  uint32_t RC[2] = {0, 0};
  int stage = 0;
#pragma unroll
  for (size_t ptr = this_ptr; ptr < next_ptr; ptr++) {

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

    __syncthreads();

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
        &A_smem[stage][tileIdY][tileIdX][lane_id % 16][(lane_id / 16) * 8]);
    LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                A_smem_lane_addr);

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
        &B_smem[stage][tileIdY][tileIdX][lane_id % 8][((lane_id / 8) % 2) * 8]);
    LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

    HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
              RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

    __syncthreads();

    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    if (stage_ptr < next_ptr) {

      size_t i = bcsrColIdxA[stage_ptr] / MMA_K;
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      cuda::memcpy_async(
          ((int4 *)(&A_smem[stage][tileIdY][tileIdX][lane_id / 2][0]) +
           lane_id % 2),
          (((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * BLOCKSIZE * MMA_K *
                                      BLOCKSIZE +
                                  (tileIdY * MMA_K * MMA_M * BLOCKSIZE +
                                   tileIdX * MMA_M * MMA_K) +
                                  (lane_id / 2) * MMA_K]) +
            lane_id % 2)),
          sizeof(int4), pipe);

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        // cuda::memcpy_async(
        //     ((int4 *)(&B_smem[stage][tileIdY][tileIdX][lane_id / 2][0]) +
        //      lane_id % 2),
        //     ((int4 *)(&B[i * MMA_K * BLOCKSIZE + tileIdX * MMA_K +
        //                  (warp_col + lane_id / 2) * K]) +
        //      lane_id % 2),
        //     sizeof(int4), pipe);
        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][tileIdY][tileIdX][lane_id / 2][0]) +
             lane_id % 2),
            ((int4 *)(&B[i * MMA_K * BLOCKSIZE + tileIdX * MMA_K +
                         (warp_col + tileIdY * WARP_SIZE + lane_id / 2) * K]) +
             lane_id % 2),
            sizeof(int4), pipe);
      }
    }

    pipe.producer_commit();

    stage = (stage + 1) % NUM_STAGES;
  }

  *((uint32_t *)(&C_smem[tileIdY][tileIdX][lane_id / 4][0]) + lane_id % 4) =
      RC[0];
  *((uint32_t *)(&C_smem[tileIdY][tileIdX][lane_id / 4 + 8][0]) + lane_id % 4) =
      RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + (tileIdY * MMA_M) + lane_id) * N + warp_col +
                  tileIdX * MMA_N])) =
        *((int4 *)(&C_smem[tileIdY][tileIdX][lane_id][0]));
  }
}

void mmaOBTKernel_tiled(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA,
                        half *B, half *C, size_t M, size_t N, size_t K,
                        size_t nonzeroBlocks, int *blockInfo,
                        int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE * BLOCKSIZE * BLOCKSIZE);
  dim3 grid(div_ceil(N, MMA_N * BLOCKSIZE), div_ceil(M, MMA_M * BLOCKSIZE));

  mmaOBTKernelSparse_tiled<<<grid, block>>>(
      bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, B, C, M, N, K, nonzeroBlocks,
      blockInfo, relativeBlockIndexMapping);
}