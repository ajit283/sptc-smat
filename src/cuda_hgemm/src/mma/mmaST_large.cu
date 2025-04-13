

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
#define MMA_K 32

#define WARP_SIZE 32

// 1 - 2:4 sparse block
// 2 - dense block

__global__ void preprocessing_mmaSTKernelSparse_large(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping) {
  int PRINT_THREAD_ID = 11;
  // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "got here 23423 \n");
  // M = M * 2;
  // N = N * 2;
  // mmaSTKernel
  const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K - 1) / (MMA_K);

  size_t blockIndex = blockRow * colRegions + blockCol;

  // print M
  // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "M: %d\n", M);

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  // __shared__ half C_smem[MMA_M][MMA_N];

  uint32_t RC[2] = {0, 0};

  // print K_tiles
  // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "K_tiles: %d\n", K_tiles);

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    size_t blockIndex = blockRow * colRegions + i;
    int sparsityInfo = blockInfo[blockIndex];
    if (blockInfo[blockIndex] == 0) {
      continue;
    }
    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    if (sparsityInfo == 2) {

    }

    else if (sparsityInfo == 1) {

      __shared__ half A_smem[MMA_M][MMA_K / 2];
      __shared__ half B_smem[MMA_N][MMA_K];
      __shared__ char Meta_smem[MMA_M][MMA_K / 8];

      half *src = // length 16
          ((half *)((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                           (lane_id / 2) * MMA_K]) +
                    lane_id % 2));

      half src_sparse[MMA_K / 2 / 2];

      char *cur_meta = (Meta_smem[lane_id / 2]) + (lane_id % 2) * 2;

      cur_meta[0] = 0;
      cur_meta[1] = 0;

      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID,
      //                    "Block %d Lane %d src values: ", blockIndex,
      //                    lane_id);
      // for (int k = 0; k < MMA_K / 2; ++k) {
      //   DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "%i ", (int)src[k]);
      // }
      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "\n");

#pragma unroll
      for (int part = 0; part < 2; ++part) {
        for (int j = 0; j < 2; ++j) {
          int cur_src_sparse = 0;
          src_sparse[0 + (2 * j) + part * 4] = 0;
          src_sparse[1 + (2 * j) + part * 4] = 0;
          for (int i = 0; i < 4; ++i) {
            if (src[i + (4 * j) + part * 8] != (half)0.0f) {
              src_sparse[cur_src_sparse + (2 * j) + part * 4] =
                  src[i + (4 * j) + part * 8];
              cur_meta[part] |= i << (6 - (2 * (1 - cur_src_sparse) + (4 * j)));
              if (cur_src_sparse > 1) {
                printf("%d ", cur_src_sparse);
              }
              cur_src_sparse++;
            }
          }
        }
      }
      // Debug: Print sparse and metadata after processing
      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID,
      //                    "Block %d Lane %d src_sparse values: ", blockIndex,
      //                    lane_id);
      // for (int k = 0; k < MMA_K / 2 / 2; ++k) {
      //   DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "%i ", (int)src_sparse[k]);
      // }
      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "\n");

      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "Block %d Lane %d metadata: \n",
      //                    blockIndex, lane_id);

      // DEBUG_EXECUTE_ON_THREAD(
      //     PRINT_THREAD_ID,
      //     for (int i = 7; i >= 0; i--) { printf("%d", (cur_meta[0] >> i) &
      //     1); }

      // )

      // DEBUG_EXECUTE_ON_THREAD(
      //     PRINT_THREAD_ID, printf("|");
      //     for (int i = 7; i >= 0; i--) { printf("%d", (cur_meta[1] >> i) &
      //     1); }

      // )

      *(metadata + (relativeIndex * MMA_M * (MMA_K / 8) + lane_id)) =
          cur_meta[0];
      *(metadata + (relativeIndex * MMA_M * (MMA_K / 8) + lane_id + 1)) =
          cur_meta[1];

      *(((int4 *)(sparseMatrixA)) + relativeIndex * MMA_M * (MMA_K / 16) +
        lane_id) = *(int4 *)src_sparse;
    }
  }
}

__global__ void mmaSTKernelSparse_large(half *bcsrValuesA, char *metadata,
                                        half *sparseMatrixA, half *B, half *C,
                                        size_t M, size_t N, size_t K,
                                        size_t nonzeroBlocks, int *blockInfo,
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
  // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "K_tiles: %d\n", K_tiles);

  __shared__ half A_smem_sparse[MMA_M][MMA_K / 2];
  __shared__ half B_smem_sparse[MMA_N][MMA_K];
  __shared__ char Meta_smem_sparse[MMA_M][MMA_K / 8];

  __shared__ half A_smem[MMA_M][MMA_K];
  __shared__ half B_smem[MMA_N][MMA_K];

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    size_t blockIndex = blockRow * colRegions + i;
    int sparsityInfo = blockInfo[blockIndex];
    if (blockInfo[blockIndex] == 0) {
      continue;
    }
    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    // print RC as uint32_t
    // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "Block %d Lane %d RC: ", blockIndex,
    //  lane_id);
    for (int k = 0; k < 2; ++k) {
      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "%i ", (int)RC[k]);
    }
    // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "\n");

    if (sparsityInfo == 2) {

      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "DENSE");

      // __shared__ half C_smem[MMA_M][MMA_N];

      *((long4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
          *(((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                    (lane_id / 2) * MMA_K])) +
            lane_id % 2);

      if (lane_id < MMA_N * 2) {
        *((long4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
            *((long4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) +
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

      HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                RC[1]);

      A_smem_lane_addr = __cvta_generic_to_shared(
          &A_smem[lane_id % 16][(lane_id / 16) * 8 + 16]);
      LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

      B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[lane_id % 8][((lane_id / 8) % 2) * 8 + 16]);
      LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

      HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                RC[1]);

    }

    else if (sparsityInfo == 1) {
      // DEBUG_PRINT_THREAD(PRINT_THREAD_ID, "SPARSE");

      *((int4 *)(&A_smem_sparse[lane_id / 2][0]) + lane_id % 2) =
          *(((int4 *)(sparseMatrixA)) + relativeIndex * MMA_M * (MMA_K / 16) +
            lane_id);

      *((half *)(Meta_smem_sparse[lane_id / 2]) + (lane_id % 2)) = *(
          (half *)metadata + (relativeIndex * MMA_M * (MMA_K / 16) + lane_id));

      *((int4 *)(&B_smem_sparse[lane_id / 4][0]) + lane_id % 4) = *(
          (int4 *)(&B[i * MMA_K + (warp_col + lane_id / 4) * K]) + lane_id % 4);

      __syncthreads();

      char metadata_local[4];

      metadata_local[0] =
          (char)((Meta_smem_sparse[lane_id / 4][0 + 2 * (lane_id % 2)]));
      metadata_local[1] =
          (char)((Meta_smem_sparse[lane_id / 4][1 + 2 * (lane_id % 2)]));
      metadata_local[2] =
          (char)((Meta_smem_sparse[(lane_id / 4) + 8][0 + 2 * (lane_id % 2)]));
      metadata_local[3] =
          (char)((Meta_smem_sparse[(lane_id / 4) + 8][1 + 2 * (lane_id % 2)]));

      uint32_t RA[4];
      uint32_t RB[4];

      uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
          &A_smem_sparse[lane_id % 16][(lane_id / 16) * (MMA_K / 2 / 2)]);
      LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem_sparse[lane_id % 8][((lane_id / 8) % 2) * (MMA_K / 2)]);
      LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], B_smem_lane_addr);

      uint32_t meta_value;
      memcpy(&meta_value, metadata_local, sizeof(uint32_t));

      HMMA16832_SPARSE(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1],
                       RB[2], RB[3], RC[0], RC[1], meta_value, 0x0);
    }
  }

  *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
  *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

  __syncthreads();

  if (lane_id < MMA_M) {
    *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) =
        *((int4 *)(&C_smem[lane_id][0]));
  }
}

void mmaSTKernel_large(half *bcsrValuesA, char *metadata, half *sparseMatrixA,
                       half *B, half *C, size_t M, size_t N, size_t K,
                       size_t nonzeroBlocks, int *blockInfo,
                       int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

  mmaSTKernelSparse_large<<<grid, block>>>(
      bcsrValuesA, metadata, sparseMatrixA, B, C, M, N, K, nonzeroBlocks,
      blockInfo, relativeBlockIndexMapping);
}

void preprocessing_mmaSTKernel_large(half *bcsrValuesA, char *metadata,
                                     half *sparseMatrixA, size_t M, size_t N,
                                     size_t K, size_t nonzeroBlocks,
                                     int *blockInfo,
                                     int *relativeBlockIndexMapping) {
  dim3 block(WARP_SIZE);
  dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

  preprocessing_mmaSTKernelSparse_large<<<grid, block>>>(
      bcsrValuesA, metadata, sparseMatrixA, M, N, K, nonzeroBlocks, blockInfo,
      relativeBlockIndexMapping);
}
