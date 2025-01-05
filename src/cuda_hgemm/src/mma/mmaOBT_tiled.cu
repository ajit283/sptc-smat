#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <stdio.h>

#include "common.h"
#include "logging_cuda.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

#define NUM_STAGES 2

#define BLOCK 2

__device__ void check_nan_half2(const char *location, half2 val, int thread_id,
                                int block_id) {
  if (__hisnan(val.x) || __hisnan(val.y)) {
    // printf("WARNING: NaN detected in %s [Thread %d, Block %d]: (%f, %f)\n",
    //        location, thread_id, block_id, __half2float(val.x),
    //        __half2float(val.y));
  }
}

__global__ void mmaOBTKernelSparse_tiled(half *bcsrValuesA, int *bcsrRowPtrA,
                                         int *bcsrColIdxA, half *B, half *C,
                                         size_t M, size_t N, size_t K,
                                         size_t nonzeroBlocks, int *blockInfo,
                                         int *relativeBlockIndexMapping) {
  // mmaCBTKernel
  // const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M * BLOCK;
  const size_t warp_col = blockIdx.x * MMA_N * BLOCK;

  const size_t warp_id = threadIdx.x / WARP_SIZE;

  const size_t warp_id_y = (threadIdx.x / WARP_SIZE) / BLOCK;
  const size_t warp_id_x = (threadIdx.x / WARP_SIZE) % BLOCK;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K * BLOCK - 1) / (MMA_K * BLOCK);

  // if (warp_row >= M || warp_col >= N) {
  //   return;
  // }

  if (warp_row + warp_id_y * MMA_M >= M || warp_col + warp_id_x * MMA_N >= N) {
    return;
  }

  __shared__ half A_smem[NUM_STAGES][MMA_M * BLOCK][MMA_K * BLOCK];
  __shared__ half B_smem[NUM_STAGES][MMA_N * BLOCK][MMA_K * BLOCK];
  __shared__ half C_smem[MMA_M * BLOCK][MMA_N * BLOCK];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][2];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  // if (blockIdx.y == 0) {
  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    if (ptr < bcsrRowPtrA[blockRow + 1]) {
      size_t i = bcsrColIdxA[ptr] / (MMA_K * BLOCK);
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      if (blockIdx.y == 1 && threadIdx.x == 0) {
        printf("BlockIndex, preload: %lu\n", blockIndex);
        printf("ptr: %lu\n", ptr);
        printf("i: %lu\n", i);
        printf("blockRow: %lu\n", blockRow);
        printf("colRegions: %lu\n", colRegions);
      }

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      cuda::memcpy_async(
          ((int4 *)(&A_smem[stage][warp_id_y * MMA_M + (lane_id / 2)]
                           [warp_id_x * MMA_K]) +
           lane_id % 2),
          (((int4 *)(&bcsrValuesA
                         [(relativeIndex)*MMA_M * MMA_K * BLOCK * BLOCK +
                          warp_id * MMA_M * MMA_K + (lane_id / 2) * MMA_K]) +
            lane_id % 2)),
          sizeof(int4), pipe);

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][warp_id_y * MMA_N + (lane_id / 2)]
                             [warp_id_x * MMA_K]) +
             lane_id % 2),
            ((int4 *)(&B[i * MMA_K * BLOCK + (warp_id_x)*MMA_K +
                         (warp_col + warp_id_y * MMA_N + lane_id / 2) * K]) +
             lane_id % 2),
            sizeof(int4), pipe);
      }

      pipe.producer_commit();
    }
    // }
  }

  uint32_t RC[2] = {0, 0};
  int stage = 0;
  // if (threadIdx.x == 0 && threadIdx.y == 0) {
  //   printf("Block %d,%d: rowPtr[%d]=%d, rowPtr[%d+1]=%d\n", blockIdx.x,
  //          blockIdx.y, (int)blockRow, bcsrRowPtrA[blockRow], (int)blockRow,
  //          bcsrRowPtrA[blockRow + 1]);
  // }

  // // print entire bcsrRowPtrA
  // printf("bcsrRowPtrA: ");
  // for (int i = 0; i < 3; i++) {
  //   DEBUG_EXECUTE_ON_THREAD(0, printf("%d ", bcsrRowPtrA[i]););
  // }
  // DEBUG_EXECUTE_ON_THREAD(0, printf("\n"););

#pragma unroll
  for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1];
       ptr++) {

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

    __syncthreads();

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
        &A_smem[stage][warp_id_y * MMA_M + (lane_id % 16)]
               [warp_id_x * MMA_K + (lane_id / 16) * 8]);
    LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                A_smem_lane_addr);

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
        &B_smem[stage][warp_id_y * MMA_N + lane_id % 8]
               [warp_id_x * MMA_K + ((lane_id / 8) % 2) * 8]);
    LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

    {
      half2 *rc0_ptr = reinterpret_cast<half2 *>(&RC[0]);
      half2 *rc1_ptr = reinterpret_cast<half2 *>(&RC[1]);

      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        half2 *rc0_ptr = reinterpret_cast<half2 *>(&RC[0]);
        half2 *rc1_ptr = reinterpret_cast<half2 *>(&RC[1]);
        printf("RC values (before): (%f,%f) (%f,%f)\n",
               __half2float(rc0_ptr->x), __half2float(rc0_ptr->y),
               __half2float(rc1_ptr->x), __half2float(rc1_ptr->y));
      }
    }

    HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
              RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

    half2 *rc0_ptr = reinterpret_cast<half2 *>(&RC[0]);
    half2 *rc1_ptr = reinterpret_cast<half2 *>(&RC[1]);

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      half2 *rc0_ptr = reinterpret_cast<half2 *>(&RC[0]);
      half2 *rc1_ptr = reinterpret_cast<half2 *>(&RC[1]);
      printf("RC values: (%f,%f) (%f,%f)\n", __half2float(rc0_ptr->x),
             __half2float(rc0_ptr->y), __half2float(rc1_ptr->x),
             __half2float(rc1_ptr->y));
    }

    __syncthreads();

    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    if (stage_ptr < bcsrRowPtrA[blockRow + 1]) {

      size_t i = bcsrColIdxA[stage_ptr] / MMA_K;
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      DEBUG_EXECUTE_ON_THREAD(0, printf("BlockIndex: %lu\n", blockIndex););

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      // cuda::memcpy_async(
      //     ((int4 *)(&A_smem[stage][lane_id / 2][0]) + lane_id % 2),
      //     (((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
      //                             (lane_id / 2) * MMA_K]) +
      //       lane_id % 2)),
      //     sizeof(int4), pipe);

      // // For matrix B
      // if (lane_id < MMA_N * 2) { // Original condition preserved
      //   cuda::memcpy_async(
      //       ((int4 *)(&B_smem[stage][lane_id / 2][0]) + lane_id % 2),
      //       ((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
      //        lane_id % 2),
      //       sizeof(int4), pipe);
      // }

      cuda::memcpy_async(
          ((int4 *)(&A_smem[stage][warp_id_y * MMA_M + (lane_id / 2)]
                           [warp_id_x * MMA_K]) +
           lane_id % 2),
          (((int4 *)(&bcsrValuesA
                         [(relativeIndex)*MMA_M * MMA_K * BLOCK * BLOCK +
                          warp_id * MMA_M * MMA_K + (lane_id / 2) * MMA_K]) +
            lane_id % 2)),
          sizeof(int4), pipe);

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][warp_id_y * MMA_N + (lane_id / 2)]
                             [warp_id_x * MMA_K]) +
             lane_id % 2),
            ((int4 *)(&B[i * MMA_K * BLOCK + (warp_id_x)*MMA_K +
                         (warp_col + warp_id_y * MMA_N + lane_id / 2) * K]) +
             lane_id % 2),
            sizeof(int4), pipe);
      }
    }

    pipe.producer_commit();

    stage = (stage + 1) % NUM_STAGES;
  }

  *((uint32_t
         *)(&C_smem[(warp_id_y * MMA_M) + lane_id / 4][warp_id_x * MMA_N]) +
    lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[(warp_id_y * MMA_M) + (lane_id / 4 + 8)]
                        [warp_id_x * MMA_N]) +
    lane_id % 4) = RC[1];

  __syncthreads();
  // figure this out

  // const size_t warp_row_c = blockIdx.y * MMA_M * BLOCK;
  // const size_t warp_col_c = blockIdx.x * MMA_N * BLOCK;
  // print warp_col_c with thread
  DEBUG_EXECUTE_ON_THREAD(0,
                          printf("Warp col, initially: %d\n", (int)warp_col););

  // for (int bl_y = 0; bl_y < BLOCK; bl_y++) {
  //   for (int bl_x = 0; bl_x < BLOCK; bl_x++) {
  if (lane_id < MMA_M) {

    // if (warp_row_c + bl_y * MMA_M + lane_id >= M ||
    //     warp_col_c + bl_x * MMA_N + 8 > N) {
    //   continue; // Skip this block
    // }
    // // Get pointers to the source and destination data
    // const half2 *src_ptr = reinterpret_cast<const half2 *>(
    //     &C_smem[(bl_y * MMA_M) + lane_id][bl_x * MMA_N]);

    const half2 *src_ptr = reinterpret_cast<const half2 *>(
        &C_smem[(warp_id_y * MMA_M) + lane_id][warp_id_x * MMA_N]);

    // half2 *dst_ptr = reinterpret_cast<half2 *>(
    //     &C[(warp_row_c + lane_id) * N + warp_col_c]);
    // half2 *dst_ptr = reinterpret_cast<half2 *>(
    //     &C[(warp_row_c + bl_y * MMA_M + lane_id) * N + warp_col_c +
    //        bl_x * MMA_N]);
    // Load values from shared memory
    half2 val0 = src_ptr[0];
    half2 val1 = src_ptr[1];
    half2 val2 = src_ptr[2];
    half2 val3 = src_ptr[3];

    DEBUG_EXECUTE_ON_THREAD(
        0, printf("val0: %f %f\n", __half2float(val0.x), __half2float(val0.y));
        printf("val1: %f %f\n", __half2float(val1.x), __half2float(val1.y));
        printf("val2: %f %f\n", __half2float(val2.x), __half2float(val2.y));
        printf("val3: %f %f\n", __half2float(val3.x), __half2float(val3.y)););

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 64 &&
        threadIdx.y == 0) {
      printf("val0 (lower): %f %f\n", __half2float(val0.x),
             __half2float(val0.y));
      printf("val1: %f %f\n", __half2float(val1.x), __half2float(val1.y));
      printf("val2: %f %f\n", __half2float(val2.x), __half2float(val2.y));
      printf("val3: %f %f\n", __half2float(val3.x), __half2float(val3.y));
    }

    // print warp col
    DEBUG_EXECUTE_ON_THREAD(0, printf("Warp col: %d\n", (int)warp_col););

    DEBUG_EXECUTE_ON_THREAD(
        0, printf("Thread info: blockIdx=(%d,%d), threadIdx=%d, "
                  "lane_id=%d, warp_id_x=%d, warp_id_y=%d\n",
                  blockIdx.x, blockIdx.y, threadIdx.x, (int)lane_id,
                  (int)warp_id_x, (int)warp_id_y);
        printf("Output indices: warp_row_c=%lu, lane_id=%d, "
               "N=%lu, warp_col_c=%d\n",
               (unsigned long)warp_row, (int)lane_id, (unsigned long)N,
               (int)warp_col););
    DEBUG_EXECUTE_ON_THREAD(0, printf("Warp col: %lu\n", warp_col););

    // if (warp_row >= 32) {
    //   // print C_smem[(warp_id_y * MMA_M) + lane_id][warp_id_x * MMA_N]
    //   printf("C_smem (warp_row: %lu) (warp_id_y: %lu): %f \n", warp_row,
    //          warp_id_y,
    //          __half2float(
    //              C_smem[(warp_id_y * MMA_M) + lane_id][warp_id_x *
    //              MMA_N]));

    //   // print entire C_smem
    //   // for (int i = 0; i < MMA_M * BLOCK; i++) {
    //   //   for (int j = 0; j < MMA_N * BLOCK; j++) {
    //   //     printf("%f ", __half2float(C_smem[i][j]));
    //   //   }
    //   //   printf("\n");
    //   // }
    // }

    *((int4 *)(&C[(warp_row + warp_id_y * MMA_M + lane_id) * N + warp_col +
                  warp_id_x * MMA_N])) =
        *((int4 *)(&C_smem[(warp_id_y * MMA_M) + lane_id][warp_id_x * MMA_N]));
  }
}

void mmaOBTKernel_tiled(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA,
                        half *B, half *C, size_t M, size_t N, size_t K,
                        size_t nonzeroBlocks, int *blockInfo,
                        int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE * BLOCK * BLOCK);
  dim3 grid(div_ceil(N, MMA_N * BLOCK), div_ceil(M, MMA_M * BLOCK));

  mmaOBTKernelSparse_tiled<<<grid, block>>>(
      bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, B, C, M, N, K, nonzeroBlocks,
      blockInfo, relativeBlockIndexMapping);
}