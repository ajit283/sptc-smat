#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <stdio.h>

#include "common.h"
#include "logging_cuda.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define WARP_SIZE 32

#define NUM_STAGES 2

#define BLOCK 2

#define ALIGNMENT_OFFSET 0

__global__ void preprocessing_mmaOBTSKernelSparse_tiled_large(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping, int *tileInfo) {

  const size_t warp_id = threadIdx.x / WARP_SIZE;
  const size_t warp_id_y = (threadIdx.x / WARP_SIZE) / BLOCK;
  const size_t warp_id_x = (threadIdx.x / WARP_SIZE) % BLOCK;
  const size_t lane_id = threadIdx.x % WARP_SIZE;

  const size_t K_tiles = div_ceil(K, MMA_K * BLOCK);

  const size_t blockRow = blockIdx.y;
  const size_t blockCol = blockIdx.x;

  const size_t colRegions = (K + (MMA_K * BLOCK) - 1) / (MMA_K * BLOCK);
  const size_t blockIndexBase = blockRow * colRegions + blockCol;

#pragma unroll
  for (size_t i = 0; i < K_tiles; ++i) {
    size_t blockIndex = blockRow * colRegions * BLOCK * BLOCK +
                        i * BLOCK * BLOCK + warp_id_y * BLOCK + warp_id_x;
    int sparsityInfo = tileInfo[blockIndex];

    if (sparsityInfo == 0) {
      continue;
    }
    size_t relativeIndex = relativeBlockIndexMapping[blockRow * colRegions + i];

    if (sparsityInfo == 2) {
      half *src = // length 16
          ((half *)((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K *
                                               BLOCK * BLOCK +
                                           warp_id * MMA_K * MMA_M +
                                           (lane_id / 2) * MMA_K]) +
                    lane_id % 2));

      *(((long4 *)(sparseMatrixA)) +
        relativeIndex * MMA_M * (MMA_K / 16) * BLOCK * BLOCK +
        warp_id * MMA_M * (MMA_K / 16) + lane_id) = *(long4 *)src;
    }

    else if (sparsityInfo == 1) {

      printf("sprasity 1");

      __shared__ half A_smem[MMA_M][MMA_K / 2];
      __shared__ half B_smem[MMA_N][MMA_K];
      __shared__ char Meta_smem[BLOCK][BLOCK][MMA_M][MMA_K / 8];

      half *src = // length 16
          ((half *)((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K *
                                               BLOCK * BLOCK +
                                           warp_id * MMA_K * MMA_M +
                                           (lane_id / 2) * MMA_K]) +
                    lane_id % 2));

      half src_sparse[MMA_K / 2]; // padding to account for dense

      memset(src_sparse, 0, sizeof(half) * MMA_K / 2);

      char *cur_meta =
          (Meta_smem[warp_id_y][warp_id_x][lane_id / 2]) + (lane_id % 2) * 2;

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

      *(metadata + (relativeIndex * BLOCK * BLOCK * MMA_M * (MMA_K / 8) +
                    warp_id * MMA_M * (MMA_K / 8) + lane_id)) = cur_meta[0];
      *(metadata + (relativeIndex * BLOCK * BLOCK * MMA_M * (MMA_K / 8) +
                    warp_id * MMA_M * (MMA_K / 8) + lane_id + 1)) = cur_meta[1];

      *(((long4 *)(sparseMatrixA)) +
        relativeIndex * MMA_M * (MMA_K / 16) * BLOCK * BLOCK +
        warp_id * MMA_M * (MMA_K / 16) + lane_id) = *(long4 *)src_sparse;
    }
  }
}

__global__ void mmaOBTSKernelSparse_tiled_large(
    half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, char *metadata,
    half *sparseMatrixA, half *B, half *C, size_t M, size_t N, size_t K,
    size_t nonzeroBlocks, int *blockInfo, int *relativeBlockIndexMapping,
    int *tileInfo) {

  // printf("LAUNCHING SPARSE TILED LARGE\n");

  const size_t warp_row = blockIdx.y * MMA_M * BLOCK;
  const size_t warp_col = blockIdx.x * MMA_N * BLOCK;

  const size_t warp_id = threadIdx.x / WARP_SIZE;
  const size_t warp_id_y = (threadIdx.x / WARP_SIZE) / BLOCK;
  const size_t warp_id_x = (threadIdx.x / WARP_SIZE) % BLOCK;
  const size_t lane_id = threadIdx.x % WARP_SIZE;

  const size_t blockRow = blockIdx.y;
  const size_t blockCol = blockIdx.x;

  if (warp_row + warp_id_y * MMA_M >= M || warp_col + warp_id_x * MMA_N >= N) {
    return;
  }

  const size_t colRegions = (K + MMA_K * BLOCK - 1) / (MMA_K * BLOCK);

  const size_t K_tiles = div_ceil(K, MMA_K);

  if (warp_row >= M || warp_col >= N) {
    return;
  }

  __shared__ half A_smem[NUM_STAGES][BLOCK][BLOCK][MMA_M][MMA_K];
  __shared__ half B_smem[NUM_STAGES][BLOCK][BLOCK][MMA_N][MMA_K];
  __shared__ half C_smem[BLOCK][BLOCK][MMA_M][MMA_N];

  memset(C_smem, 0, sizeof(half) * BLOCK * BLOCK * MMA_M * MMA_N);
  memset(A_smem, 0, sizeof(half) * NUM_STAGES * BLOCK * BLOCK * MMA_M * MMA_K);
  memset(B_smem, 0, sizeof(half) * NUM_STAGES * BLOCK * BLOCK * MMA_N * MMA_K);

  // __shared__ half A_smem_sparse[NUM_STAGES][MMA_M][MMA_K / 2];
  // __shared__ half B_smem_sparse[NUM_STAGES][MMA_N][MMA_K];
  __shared__ char Meta_smem_sparse[NUM_STAGES][BLOCK][BLOCK][MMA_M][MMA_K / 8];

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][4];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  auto loadStages = [&] __device__(size_t stage_ptr, int stage) {
    if (stage_ptr < bcsrRowPtrA[blockRow + 1]) {

      size_t i = bcsrColIdxA[stage_ptr] / (MMA_K * BLOCK);
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      size_t tileIndex = blockRow * colRegions * BLOCK * BLOCK +
                         i * BLOCK * BLOCK + warp_id_y * BLOCK + warp_id_x;
      int sparsityInfo = tileInfo[tileIndex];

      if (sparsityInfo == 2) {

        cuda::memcpy_async(
            ((long4 *)(&A_smem[stage][warp_id_y][warp_id_x][lane_id / 2][0]) +
             lane_id % 2),
            (((long4 *)(sparseMatrixA)) +
             relativeIndex * MMA_M * (MMA_K / 16) * BLOCK * BLOCK +
             warp_id * MMA_M * (MMA_K / 16) + lane_id),
            sizeof(long4), pipe);

        // For matrix B
        if (lane_id < MMA_N * 2) { // Original condition preserved
          cuda::memcpy_async(
              ((long4 *)(&B_smem[stage][warp_id_y][warp_id_x][(lane_id / 2)]
                                [(ALIGNMENT_OFFSET)]) +
               lane_id % 2),
              ((long4 *)(&B[i * MMA_K * BLOCK + (warp_id_x)*MMA_K +
                            (warp_col + warp_id_y * MMA_N + lane_id / 2) * K]) +
               lane_id % 2),
              sizeof(long4), pipe);
        }

      } else if (sparsityInfo == 1) {

        printf("SPARSEEEE");

        cuda::memcpy_async(
            ((int4 *)(&A_smem[stage][warp_id_y][warp_id_x][lane_id / 2][0]) +
             lane_id % 2),
            (((int4 *)(sparseMatrixA)) + relativeIndex * MMA_M * (MMA_K / 16) +
             lane_id),
            sizeof(int4), pipe);

        cuda::memcpy_async(((half *)(Meta_smem_sparse[stage][warp_id_y]
                                                     [warp_id_x][lane_id / 2]) +
                            (lane_id % 2)),
                           ((half *)metadata +
                            (relativeIndex * MMA_M * (MMA_K / 8) + lane_id)),
                           sizeof(half), pipe);

        // cuda::memcpy_async(((half *)(Meta_smem_sparse[stage][warp_id_y]
        //                                              [warp_id_x][lane_id /
        //                                              2]) +
        //                     (lane_id % 2)),
        //                    ((half *)metadata +
        //                     (relativeIndex * MMA_M * (MMA_K / 8) + lane_id)),
        //                    sizeof(half), pipe);

        // *((half *)(Meta_smem_sparse[stage][warp_id_y][warp_id_x][lane_id /
        // 2]) +
        //   (lane_id % 2)) = *((half *)metadata +
        //                      (relativeIndex * MMA_M * (MMA_K / 8) +
        //                      lane_id));

        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][lane_id / 4][0]) + lane_id % 4),
            ((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 4) * K]) +
             lane_id % 4),
            sizeof(int4), pipe);
      }
    }
  };

  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    loadStages(ptr, stage);
  }

  uint32_t RC[2] = {0, 0};
  int stage = 0;
#pragma unroll
  for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1];
       ptr++) {

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
    size_t i = bcsrColIdxA[ptr] / (MMA_K * BLOCK);
    // skip empty block
    size_t blockIndex = blockRow * colRegions + i;

    size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

    size_t A_size = MMA_M * MMA_K * sizeof(half);
    size_t B_size = MMA_N * MMA_K * sizeof(half);

    // int sparsityInfo = blockInfo[blockIndex];
    size_t tileIndex = blockRow * colRegions * BLOCK * BLOCK +
                       i * BLOCK * BLOCK + warp_id_y * BLOCK + warp_id_x;
    int sparsityInfo = tileInfo[tileIndex];

    __syncthreads();

    if (sparsityInfo == 2) {

      uint32_t A_smem_lane_addr =
          __cvta_generic_to_shared(&A_smem[stage][warp_id_y][warp_id_x]
                                          [lane_id % 16][(lane_id / 16) * 8]);
      LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                  A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[stage][warp_id_y][warp_id_x][lane_id % 8]
                 [((lane_id / 8) % 2) * 8]);
      LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

      HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
                RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

      A_smem_lane_addr = __cvta_generic_to_shared(
          &A_smem[stage][warp_id_y][warp_id_x][lane_id % 16]
                 [(lane_id / 16) * 8 + 16]);
      LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                  A_smem_lane_addr);

      B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[stage][warp_id_y][warp_id_x][lane_id % 8]
                 [((lane_id / 8) % 2) * 8 + 16]);
      LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

      HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
                RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);

    } else if (sparsityInfo == 1) {

      uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
          &A_smem[stage][warp_id_y][warp_id_x][lane_id % 16]
                 [(lane_id / 16) * (MMA_K / 2 / 2)]);
      LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                  A_smem_lane_addr);

      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[stage][warp_id_y][warp_id_x][lane_id % 8]
                 [((lane_id / 8) % 2) * (MMA_K / 2)]);
      LDMATRIX_X4(RB[stage][0], RB[stage][1], RB[stage][2], RB[stage][3],
                  B_smem_lane_addr);

      char metadata_local[4];

      metadata_local[0] =
          (char)((Meta_smem_sparse[stage][warp_id_y][warp_id_x][lane_id / 4]
                                  [0 + 2 * (lane_id % 2)]));
      metadata_local[1] =
          (char)((Meta_smem_sparse[stage][warp_id_y][warp_id_x][lane_id / 4]
                                  [1 + 2 * (lane_id % 2)]));
      metadata_local[2] =
          (char)((Meta_smem_sparse[stage][warp_id_y][warp_id_x]
                                  [(lane_id / 4) + 8][0 + 2 * (lane_id % 2)]));
      metadata_local[3] =
          (char)((Meta_smem_sparse[stage][warp_id_y][warp_id_x]
                                  [(lane_id / 4) + 8][1 + 2 * (lane_id % 2)]));

      uint32_t meta_value;
      memcpy(&meta_value, metadata_local, sizeof(uint32_t));

      // HMMA16816_SPARSE(RC[0], RC[1], RA[stage][0], RA[stage][1],
      // RB[stage][0],
      //                  RB[stage][1], RC[0], RC[1], meta_value, 0x0);

      HMMA16832_SPARSE(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
                       RA[stage][3], RB[stage][0], RB[stage][1], RB[stage][2],
                       RB[stage][3], RC[0], RC[1], meta_value, 0x0);
    }

    __syncthreads();

    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    loadStages(stage_ptr, stage);

    pipe.producer_commit();

    stage = (stage + 1) % NUM_STAGES;
  }

  *((uint32_t *)(&C_smem[warp_id_y][warp_id_x][lane_id / 4][0]) + lane_id % 4) =
      RC[0];
  *((uint32_t *)(&C_smem[warp_id_y][warp_id_x][(lane_id / 4 + 8)][0]) +
    lane_id % 4) = RC[1];
  __syncthreads();

  if (lane_id < MMA_M) {

    *((int4 *)(&C[(warp_row + warp_id_y * MMA_M + lane_id) * N + warp_col +
                  warp_id_x * MMA_N])) =
        *((int4 *)(&C_smem[warp_id_y][warp_id_x][lane_id][0]));
  }
}

void preprocessing_mmaOBTSKernel_tiled_large(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping, int *tileInfo) {
  // Configure grid and block dimensions for tiling.
  // Each CUDA block covers a tile of (MMA_N*BLOCK) columns x
  // (MMA_M*BLOCK) rows.
  dim3 block(WARP_SIZE * BLOCK * BLOCK);
  dim3 grid(div_ceil(N, MMA_N * BLOCK), div_ceil(M, MMA_M * BLOCK));

  preprocessing_mmaOBTSKernelSparse_tiled_large<<<grid, block>>>(
      bcsrValuesA, metadata, sparseMatrixA, M, N, K, nonzeroBlocks, blockInfo,
      relativeBlockIndexMapping, tileInfo);
}

void mmaOBTSKernel_tiled_large(half *bcsrValuesA, int *bcsrRowPtrA,
                               int *bcsrColIdxA, char *metadata,
                               half *sparseMatrixA, half *B, half *C, size_t M,
                               size_t N, size_t K, size_t nonzeroBlocks,
                               int *blockInfo, int *relativeBlockIndexMapping,
                               int *tileInfo) {

  dim3 block(WARP_SIZE * BLOCK * BLOCK);
  dim3 grid(div_ceil(N, MMA_N * BLOCK), div_ceil(M, MMA_M * BLOCK));

  // char *test_md;

  // cudaMalloc(&test_md, MMA_M * MMA_K / 16 * 8192 * sizeof(char));

  mmaOBTSKernelSparse_tiled_large<<<grid, block>>>(
      bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, metadata, sparseMatrixA, B, C, M,
      N, K, nonzeroBlocks, blockInfo, relativeBlockIndexMapping, tileInfo);

  // preprocessing_mmaOBTSKernelSparse_tiled_large<<<grid, block>>>(
  //     bcsrValuesA, metadata, sparseMatrixA, M, N, K, nonzeroBlocks,
  //     blockInfo, relativeBlockIndexMapping, tileInfo);
}