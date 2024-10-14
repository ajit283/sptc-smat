

// #include <__clang_cuda_builtin_vars.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <stdio.h>

#include "common.h"
#include "logging_cuda.h"
#include "ptx.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

// 1 - 2:4 sparse block
// 2 - dense block

__global__ void mmaSTKernelSparse(half *bcsrValuesA, half *B, half *C, size_t M,
                                  size_t N, size_t K, size_t nonzeroBlocks,
                                  int *blockInfo,
                                  int *relativeBlockIndexMapping) {
  int PRINT_THREAD_ID = 11;
  // mmaSTKernel
  const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K - 1) / (MMA_K);

  size_t blockIndex = blockRow * colRegions + blockCol;

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  __shared__ half C_smem[MMA_M][MMA_N];

  uint32_t RC[2] = {0, 0};

  // print K_tiles
  DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "K_tiles: %d\n", K_tiles);

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    // skip empty block

    // if (sparsityInfo == 0) {
    //   printf("zero block");
    // } else if (sparsityInfo == 1) {
    //   printf("sparse block");
    // } else if (sparsityInfo == 2) {
    //   printf("dense block");
    // } else {
    //   printf("unknown block");
    // }
    size_t blockIndex = blockRow * colRegions + i;
    int sparsityInfo = blockInfo[blockIndex];
    if (blockInfo[blockIndex] == 0) {
      continue;
    }
    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    // _shared__ half C_smem[MMA_M][MMA_N];
    // if (sparsityInfo == 2) {

    // if (true) {
    if (sparsityInfo == 2) {

      __shared__ half A_smem[MMA_M][MMA_K];
      __shared__ half B_smem[MMA_N][MMA_K];
      // __shared__ half C_smem[MMA_M][MMA_N];

      *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
          *((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                  (lane_id / 2) * MMA_K]) +
            lane_id % 2);

      // print matrix

      if (lane_id < MMA_N * 2) {
        *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
              lane_id % 2);
      }

      __syncthreads();

      uint32_t RA[4];
      uint32_t RB[2];

      uint32_t A_smem_lane_addr =
          __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
      LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
      LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

      if (blockIdx.x == 0 && blockIdx.y == 0 &&
          threadIdx.x == PRINT_THREAD_ID) {
      }

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix part \n");
          // print RA
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RA) + i)));
          } printf("\n");
          // print RB
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RB) + i)));
          } printf("\n");
          // print RC
          for (int i = 0; i < 4;
               i++) { printf("%f ", __half2float(*(((half *)RC) + i))); })

      HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                RC[1]);

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix part after \n");
          // print RA
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RA) + i)));
          } printf("\n");
          // print RB
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RB) + i)));
          } printf("\n");
          // print RC
          for (int i = 0; i < 4;
               i++) { printf("%f ", __half2float(*(((half *)RC) + i))); })
    }

    else if (sparsityInfo == 1) {

      //------------------------------------------------------------------------------
      __shared__ half A_smem_test[MMA_M][MMA_K];

      *((int4 *)(&A_smem_test[lane_id / 2][0]) + lane_id % 2) =
          *((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                  (lane_id / 2) * MMA_K]) +
            lane_id % 2);

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix \n");
          for (int i = 0; i < MMA_M; i++) {
            for (int j = 0; j < MMA_K; j++) {
              printf("%f ", __half2float(A_smem_test[i][j]));
            }
            printf("\n");
          }

          printf("\n\n\n");
          printf("end print matrix \n");)

      // -----------------------------------------------------------------------------

      __shared__ half A_smem[MMA_M][MMA_K / 2];
      __shared__ half B_smem[MMA_N][MMA_K];
      // __shared__ half C_smem[MMA_M][MMA_N];
      __shared__ char Meta_smem[MMA_M][MMA_K / 8];

      half *src = // length 8
          ((half *)((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                          (lane_id / 2) * MMA_K]) +
                    lane_id % 2));

      half src_sparse[4];

      char *cur_meta = (Meta_smem[lane_id / 2]) + (lane_id % 2);

      *cur_meta = 0;

      for (int j = 0; j < 2; ++j) {
        int cur_src_sparse = 0;
        src_sparse[0 + (2 * j)] = 0;
        src_sparse[1 + (2 * j)] = 0;
        for (int i = 0; i < 4; ++i) {
          if (src[i + (4 * j)] != (half)0.0f) {
            src_sparse[cur_src_sparse + (2 * j)] = src[i + (4 * j)];
            *cur_meta |= i << (6 - (2 * (1 - cur_src_sparse) + (4 * j)));
            // *cur_meta |= i << (2 * cur_src_sparse + (4 * j));
            cur_src_sparse++;
          }
        }
        // all zeroes
        if (cur_src_sparse == 0) {
          *cur_meta |= 0b0100 << (4 * j);
          // *cur_meta |= 0b0100 << (6 - 4 * j);
        }
      }

      *((int2 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
          *((int2 *)src_sparse);

      if (lane_id < MMA_N * 2) {
        *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
              lane_id % 2);
      }

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix sparse \n");
          for (int i = 0; i < MMA_M; i++) {
            for (int j = 0; j < MMA_K / 2; j++) {
              printf("%f ", __half2float(A_smem[i][j]));
            }
            printf("\n");
          }

          printf("\n\n\n");
          printf("end print matrix sparse \n");)

      char metadata[4];

      // metadata[0] = *cur_meta;
      metadata[0] = (char)((Meta_smem[lane_id / 4][0]));
      metadata[1] = (char)((Meta_smem[lane_id / 4][1]));
      metadata[2] = (char)((Meta_smem[(lane_id / 4) + 8][0]));
      metadata[3] = (char)((Meta_smem[(lane_id / 4) + 8][1]));

      // metadata[2] = *(cur_meta + (2 * 8));
      // metadata[2] = *(cur_meta + (2 * 8) + 1);

      // print metadata[0] as bits

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, // print lane id
          printf("lane id: %d\n", lane_id);
          printf("begin print metadata as bits \n");
          for (int i = 0; i < 4; i++) {
            for (int bit = 7; bit >= 0; bit--) {
              printf("%d", (metadata[i] >> bit) & 1);
            }
            printf(" "); // Space between each char
          } printf("\n");
          printf("end print metadata as bits \n");)

      uint32_t RA[2];
      uint32_t RB[2];

      uint32_t A_smem_lane_addr =
          __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 4]);
      LDMATRIX_X2(RA[0], RA[1], A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
      LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix part \n");
          // print RA
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RA) + i)));
          } printf("\n");
          // print RB
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RB) + i)));
          } printf("\n");
          // print RC
          for (int i = 0; i < 4;
               i++) { printf("%f ", __half2float(*(((half *)RC) + i))); })

      // equivalent for metadata

      uint32_t meta_value;
      memcpy(&meta_value, metadata, sizeof(uint32_t));

      HMMA16816_SPARSE(RC[0], RC[1], RA[0], RA[1], RB[0], RB[1], RC[0], RC[1],
                       meta_value, 0x0);

      DEBUG_EXECUTE_ON_THREAD(
          PRINT_THREAD_ID, printf("begin print matrix part after \n");
          // print RA
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RA) + i)));
          } printf("\n");
          // print RB
          for (int i = 0; i < 4; i++) {
            printf("%f ", __half2float(*(((half *)RB) + i)));
          } printf("\n");
          // print RC
          for (int i = 0; i < 4;
               i++) { printf("%f ", __half2float(*(((half *)RC) + i))); })
    }

    // __syncthreads();
  }

  *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4 *)(&C_smem[lane_id][0]));
  }
}

void mmaSTKernel(half *bcsrValuesA, half *B, half *C, size_t M, size_t N,
                 size_t K, size_t nonzeroBlocks, int *blockInfo,
                 int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

  mmaSTKernelSparse<<<grid, block>>>(bcsrValuesA, B, C, M, N, K, nonzeroBlocks,
                                     blockInfo, relativeBlockIndexMapping);
}
