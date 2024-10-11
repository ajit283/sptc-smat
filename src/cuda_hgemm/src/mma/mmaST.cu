

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <stdio.h>

#include "common.h"
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

  // mmaSTKernel
  const size_t K_tiles = div_ceil(K, MMA_K);

  const size_t warp_row = blockIdx.y * MMA_M;
  const size_t warp_col = blockIdx.x * MMA_N;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K - 1) / (MMA_K);

  size_t blockIndex = blockRow * colRegions + blockCol;

  int sparsityInfo = blockInfo[blockIndex];

  if (sparsityInfo == 0) {
    // printf("zero block");
  } else if (sparsityInfo == 1) {
    // printf("sparse block");
  } else if (sparsityInfo) {
    // printf("dense block");
  }

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  __shared__ half C_smem[MMA_M][MMA_N];

  uint32_t RC[2] = {0, 0};

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    // skip empty block
    size_t blockIndex = blockRow * colRegions + i;
    if (blockInfo[blockIndex] == 0) {
      continue;
    }
    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];
    // _shared__ half C_smem[MMA_M][MMA_N];
    if (sparsityInfo == 2) {
      // if (true) {

      __shared__ half A_smem[MMA_M][MMA_K];
      __shared__ half B_smem[MMA_N][MMA_K];
      // __shared__ half C_smem[MMA_M][MMA_N];

      *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
          *((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                  (lane_id / 2) * MMA_K]) +
            lane_id % 2);

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

      HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                RC[1]);
    }

    else if (sparsityInfo == 1) {

      __shared__ half A_smem[MMA_M][MMA_K / 2];
      __shared__ half B_smem[MMA_N][MMA_K];
      // __shared__ half C_smem[MMA_M][MMA_N];
      __shared__ char Meta_smem[MMA_M][MMA_K / 8];

      half2 *src =
          ((half2 *)((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K +
                                           (lane_id / 2) * MMA_K]) +
                     lane_id % 2));

      half src_sparse[4];

      char *cur_meta = (Meta_smem[lane_id / 2]) + (lane_id % 2);

      for (int i = 0; i < 4; ++i) {
        half2 pair = src[i];
        half non_zero = (pair.x != (half)0.0f) ? pair.x : pair.y;
        src_sparse[i] = non_zero;

        // Set the metadata bits
        char position = (pair.x != (half)0.0f) ? (i * 2) : (i * 2 + 1);
        *cur_meta |= (position & 0x3) << (i * 2);
      }

      *((int2 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
          *((int2 *)src_sparse);

      __syncthreads();

      char metadata[4];

      metadata[0] = *cur_meta;
      metadata[1] = *(cur_meta + 1);
      metadata[2] = *(cur_meta + (2 * 8));
      metadata[2] = *(cur_meta + (2 * 8) + 1);

      uint32_t RA[2];
      uint32_t RB[2];

      uint32_t A_smem_lane_addr =
          __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 4]);
      LDMATRIX_X2(RA[0], RA[1], A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
      LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

      // equivalent for metadata

      HMMA16816_SPARSE(RC[0], RC[1], RA[0], RA[1], RB[0], RB[1], RC[0], RC[1],
                       *(uint32_t *)metadata, 0x0);
    }

    __syncthreads();
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
