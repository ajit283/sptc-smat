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

#define BLOCK 4

#define ALIGNMENT_OFFSET 0

__global__ void preprocessing_mmaOBTSKernelSparse_tiled_large(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping, int *tileInfo) {

  const size_t warp_id = threadIdx.x / WARP_SIZE;
  const size_t warp_id_y = (threadIdx.x / WARP_SIZE) / BLOCK;
  const size_t warp_id_x = (threadIdx.x / WARP_SIZE) % BLOCK;
  const size_t lane_id = threadIdx.x % WARP_SIZE;

  const size_t blockRow = blockIdx.y;
  const size_t blockCol = blockIdx.x;

  const size_t colRegions = (K + (MMA_K * BLOCK) - 1) / (MMA_K * BLOCK);
  const size_t blockIndexBase = blockRow * colRegions + blockCol;

  // Process all subtiles in the current tile
  for (size_t subtile_y = 0; subtile_y < BLOCK; subtile_y++) {
    for (size_t subtile_x = 0; subtile_x < BLOCK; subtile_x++) {
      size_t blockIndex = blockIndexBase + subtile_y * colRegions + subtile_x;
      int sparsityInfo =
          tileInfo[blockIndex * BLOCK * BLOCK + subtile_y * BLOCK + subtile_x];

      if (sparsityInfo == 0) {
        // Zero block, skip
        continue;
      }

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];
      if (!relativeIndex)
        continue; // Skip if no mapping exists

      // Calculate offsets for this subtile
      size_t subtileOffset =
          (relativeIndex * BLOCK * BLOCK + subtile_y * BLOCK + subtile_x) *
          MMA_M * MMA_K;

      if (sparsityInfo == 1) {
        // 2:4 sparse block - needs preprocessing
        half *src =
            &bcsrValuesA[subtileOffset + (lane_id / 2) * MMA_K + (lane_id % 2)];
        half src_sparse[MMA_K / 2 / 2]; // Buffer for processed values

        char *cur_meta = (char *)(metadata + subtileOffset / 2 + lane_id);
        cur_meta[0] = 0;
        cur_meta[1] = 0;

// Process the sparse data in two parts
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
                cur_meta[part] |= i
                                  << (6 - (2 * (1 - cur_src_sparse) + (4 * j)));
                if (cur_src_sparse > 1) {
                  printf("%d ", cur_src_sparse);
                }
                cur_src_sparse++;
                // print cur_src_sparse
              }
            }
          }
        }

        // Store processed data
        // half *dest = &sparseMatrixA[subtileOffset + lane_id * (MMA_K / 16)];
        // *((int4 *)dest) = *((int4 *)src_sparse);

        // Corrected code:
        half *dest =
            &sparseMatrixA[subtileOffset + (lane_id / 2) * (MMA_K / 2) +
                           (lane_id % 2) * 8];
        *((int4 *)dest) = *((int4 *)src_sparse);

        // Store metadata
        metadata[subtileOffset / 2 + lane_id] = cur_meta[0];
        metadata[subtileOffset / 2 + lane_id + 1] = cur_meta[1];
      } else if (sparsityInfo == 2) {
        // Dense block - copy as is
        half *src =
            &bcsrValuesA[subtileOffset + (lane_id / 2) * MMA_K + (lane_id % 2)];
        half *dest = &sparseMatrixA[subtileOffset + (lane_id / 2) * MMA_K +
                                    (lane_id % 2)];

        // Copy the entire dense block
        *((long4 *)dest) = *((long4 *)src);

        // Zero out corresponding metadata
        if (lane_id < MMA_M * (MMA_K / 8)) {
          metadata[subtileOffset / 2 + lane_id] = 0;
        }
      }
    }
  }
}

__global__ void mmaOBTSKernelSparse_tiled_large(
    half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, char *metadata,
    half *sparseMatrixA, half *B, half *C, size_t M, size_t N, size_t K,
    size_t nonzeroBlocks, int *blockInfo, int *relativeBlockIndexMapping,
    int *tileInfo) {

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

  // Shared memory for loading data and calculations
  __shared__ half A_smem_sparse[NUM_STAGES][BLOCK][BLOCK][MMA_M]
                               [(MMA_K / 2 + ALIGNMENT_OFFSET)];
  __shared__ char Meta_smem_sparse[NUM_STAGES][BLOCK][BLOCK][MMA_M]
                                  [(MMA_K / 8 + ALIGNMENT_OFFSET)];
  __shared__ half
      A_smem_dense[NUM_STAGES][BLOCK][BLOCK][MMA_M][(MMA_K + ALIGNMENT_OFFSET)];
  __shared__ half
      B_smem[NUM_STAGES][BLOCK][BLOCK][MMA_N][(MMA_K + ALIGNMENT_OFFSET)];
  __shared__ half C_smem[BLOCK][BLOCK][MMA_M][MMA_N];

  // Storage for matrix multiplication registers
  uint32_t RA_sparse[NUM_STAGES][4];
  uint32_t RA_dense[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][4]; // Need 4 for sparse, 2 for dense
  uint32_t RC[2] = {0, 0};    // Accumulation registers

  // Setup pipeline for async memory operations
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  // Load initial pipeline stages
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    if (ptr < bcsrRowPtrA[blockRow + 1]) {
      size_t tile_idx = bcsrColIdxA[ptr] / (MMA_K * BLOCK);
      size_t blockIndex = blockRow * colRegions + tile_idx;
      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      if (relativeIndex != -1) {
        // Load the B matrix part (same for both dense and sparse)
        if (lane_id < MMA_N * 4) {
          cuda::memcpy_async(
              ((int4 *)(&B_smem[stage][warp_id_y][warp_id_x][lane_id / 4]
                               [ALIGNMENT_OFFSET]) +
               lane_id % 4),
              ((int4 *)(&B[tile_idx * MMA_K * BLOCK +
                           (warp_col + warp_id_x * MMA_N + lane_id / 4) * K]) +
               lane_id % 4),
              sizeof(int4), pipe);
        }

        // Get sparsity type for this subtile
        int subtile_idx = warp_id_y * BLOCK + warp_id_x;
        int sparsityInfo = tileInfo[blockIndex * BLOCK * BLOCK + subtile_idx];

        if (sparsityInfo == 1) {
          // Sparse block - load compressed values and metadata
          size_t subtileOffset =
              (relativeIndex * BLOCK * BLOCK + subtile_idx) * MMA_M * MMA_K;

          cuda::memcpy_async(
              ((int4 *)(&A_smem_sparse[stage][warp_id_y][warp_id_x][lane_id / 2]
                                      [ALIGNMENT_OFFSET]) +
               lane_id % 2),
              ((int4 *)(&sparseMatrixA[subtileOffset +
                                       (lane_id / 2) * (MMA_K / 2)])),
              sizeof(int4), pipe);

          if (lane_id < MMA_M * (MMA_K / 8)) {
            cuda::memcpy_async(
                &Meta_smem_sparse[stage][warp_id_y][warp_id_x]
                                 [lane_id / (MMA_K / 8)][lane_id % (MMA_K / 8)],
                &metadata[subtileOffset / 2 + lane_id], sizeof(char), pipe);
          }
        } else if (sparsityInfo == 2) {
          // Dense block - load full values
          size_t subtileOffset =
              (relativeIndex * BLOCK * BLOCK + subtile_idx) * MMA_M * MMA_K;

          cuda::memcpy_async(
              ((int4 *)(&A_smem_dense[stage][warp_id_y][warp_id_x][lane_id / 2]
                                     [ALIGNMENT_OFFSET]) +
               lane_id % 2),
              ((int4 *)(&sparseMatrixA[subtileOffset + (lane_id / 2) * MMA_K])),
              sizeof(int4), pipe);
        }
      }
    }
    pipe.producer_commit();
  }

  // Process blocks using the pipeline
  int stage = 0;
  for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1];
       ptr++) {
    // Wait for data to be ready
    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
    __syncthreads();

    size_t tile_idx = bcsrColIdxA[ptr] / (MMA_K * BLOCK);
    size_t blockIndex = blockRow * colRegions + tile_idx;

    // Process each subtile in the current block
    for (int subtile_y = 0; subtile_y < BLOCK; subtile_y++) {
      for (int subtile_x = 0; subtile_x < BLOCK; subtile_x++) {
        int sparsityInfo = tileInfo[blockIndex * BLOCK * BLOCK +
                                    subtile_y * BLOCK + subtile_x];

        if (sparsityInfo == 0)
          continue; // Skip zero blocks

        if (sparsityInfo == 1) {
          // Process sparse block
          uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
              &A_smem_sparse[stage][subtile_y][subtile_x][(lane_id % 16)]
                            [ALIGNMENT_OFFSET + (lane_id / 16) * 4]);
          LDMATRIX_X4(RA_sparse[stage][0], RA_sparse[stage][1],
                      RA_sparse[stage][2], RA_sparse[stage][3],
                      A_smem_lane_addr);

          uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
              &B_smem[stage][subtile_y][subtile_x][lane_id % 8]
                     [ALIGNMENT_OFFSET + ((lane_id / 8) % 4) * 8]);
          LDMATRIX_X4(RB[stage][0], RB[stage][1], RB[stage][2], RB[stage][3],
                      B_smem_lane_addr);

          // Load metadata
          char metadata_local[4];
          metadata_local[0] =
              Meta_smem_sparse[stage][subtile_y][subtile_x][lane_id / 4]
                              [0 + 2 * (lane_id % 2)];
          metadata_local[1] =
              Meta_smem_sparse[stage][subtile_y][subtile_x][lane_id / 4]
                              [1 + 2 * (lane_id % 2)];
          metadata_local[2] =
              Meta_smem_sparse[stage][subtile_y][subtile_x][(lane_id / 4) + 8]
                              [0 + 2 * (lane_id % 2)];
          metadata_local[3] =
              Meta_smem_sparse[stage][subtile_y][subtile_x][(lane_id / 4) + 8]
                              [1 + 2 * (lane_id % 2)];

          uint32_t meta_value;
          memcpy(&meta_value, metadata_local, sizeof(uint32_t));

          // Sparse matrix multiplication
          HMMA16832_SPARSE(RC[0], RC[1], RA_sparse[stage][0],
                           RA_sparse[stage][1], RA_sparse[stage][2],
                           RA_sparse[stage][3], RB[stage][0], RB[stage][1],
                           RB[stage][2], RB[stage][3], RC[0], RC[1], meta_value,
                           0x0);
        } else if (sparsityInfo == 2) {
          // Process dense block
          uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
              &A_smem_dense[stage][subtile_y][subtile_x][(lane_id % 16)]
                           [ALIGNMENT_OFFSET + (lane_id / 16) * 8]);
          LDMATRIX_X4(RA_dense[stage][0], RA_dense[stage][1],
                      RA_dense[stage][2], RA_dense[stage][3], A_smem_lane_addr);

          uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
              &B_smem[stage][subtile_y][subtile_x][lane_id % 8]
                     [ALIGNMENT_OFFSET + ((lane_id / 8) % 2) * 8]);
          LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

          // Dense matrix multiplication
          HMMA16816(RC[0], RC[1], RA_dense[stage][0], RA_dense[stage][1],
                    RA_dense[stage][2], RA_dense[stage][3], RB[stage][0],
                    RB[stage][1], RC[0], RC[1]);

          // Process second half of K dimension
          A_smem_lane_addr = __cvta_generic_to_shared(
              &A_smem_dense[stage][subtile_y][subtile_x][(lane_id % 16)]
                           [ALIGNMENT_OFFSET + (lane_id / 16) * 8 + 16]);
          LDMATRIX_X4(RA_dense[stage][0], RA_dense[stage][1],
                      RA_dense[stage][2], RA_dense[stage][3], A_smem_lane_addr);

          B_smem_lane_addr = __cvta_generic_to_shared(
              &B_smem[stage][subtile_y][subtile_x][lane_id % 8]
                     [ALIGNMENT_OFFSET + ((lane_id / 8) % 2) * 8 + 16]);
          LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

          HMMA16816(RC[0], RC[1], RA_dense[stage][0], RA_dense[stage][1],
                    RA_dense[stage][2], RA_dense[stage][3], RB[stage][0],
                    RB[stage][1], RC[0], RC[1]);
        }
      }
    }

    __syncthreads();

    // Release the consumed stage
    pipe.consumer_release();

    // Pre-load data for future stages
    pipe.producer_acquire();
    size_t stage_ptr = ptr + NUM_STAGES;
    if (stage_ptr < bcsrRowPtrA[blockRow + 1]) {
      size_t tile_idx = bcsrColIdxA[stage_ptr] / (MMA_K * BLOCK);
      size_t blockIndex = blockRow * colRegions + tile_idx;
      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      if (relativeIndex != -1) {
        // Load the B matrix part (same for both dense and sparse)
        if (lane_id < MMA_N * 4) {
          cuda::memcpy_async(
              ((int4 *)(&B_smem[stage][warp_id_y][warp_id_x][lane_id / 4]
                               [ALIGNMENT_OFFSET]) +
               lane_id % 4),
              ((int4 *)(&B[tile_idx * MMA_K * BLOCK +
                           (warp_col + warp_id_x * MMA_N + lane_id / 4) * K]) +
               lane_id % 4),
              sizeof(int4), pipe);
        }

        // Get sparsity type for this subtile
        int subtile_idx = warp_id_y * BLOCK + warp_id_x;
        int sparsityInfo = tileInfo[blockIndex * BLOCK * BLOCK + subtile_idx];

        if (sparsityInfo == 1) {
          // Sparse block - load compressed values and metadata
          size_t subtileOffset =
              (relativeIndex * BLOCK * BLOCK + subtile_idx) * MMA_M * MMA_K;

          cuda::memcpy_async(
              ((int4 *)(&A_smem_sparse[stage][warp_id_y][warp_id_x][lane_id / 2]
                                      [ALIGNMENT_OFFSET]) +
               lane_id % 2),
              ((int4 *)(&sparseMatrixA[subtileOffset +
                                       (lane_id / 2) * (MMA_K / 2)])),
              sizeof(int4), pipe);

          if (lane_id < MMA_M * (MMA_K / 8)) {
            cuda::memcpy_async(
                &Meta_smem_sparse[stage][warp_id_y][warp_id_x]
                                 [lane_id / (MMA_K / 8)][lane_id % (MMA_K / 8)],
                &metadata[subtileOffset / 2 + lane_id], sizeof(char), pipe);
          }
        } else if (sparsityInfo == 2) {
          // Dense block - load full values
          size_t subtileOffset =
              (relativeIndex * BLOCK * BLOCK + subtile_idx) * MMA_M * MMA_K;

          cuda::memcpy_async(
              ((int4 *)(&A_smem_dense[stage][warp_id_y][warp_id_x][lane_id / 2]
                                     [ALIGNMENT_OFFSET]) +
               lane_id % 2),
              ((int4 *)(&sparseMatrixA[subtileOffset + (lane_id / 2) * MMA_K])),
              sizeof(int4), pipe);
        }
      }
    }
    pipe.producer_commit();

    // Update stage for next iteration
    stage = (stage + 1) % NUM_STAGES;
  }

  // Store results to C_smem
  *((uint32_t *)(&C_smem[warp_id_y][warp_id_x][lane_id / 4][0]) + lane_id % 4) =
      RC[0];
  *((uint32_t *)(&C_smem[warp_id_y][warp_id_x][(lane_id / 4 + 8)][0]) +
    lane_id % 4) = RC[1];

  __syncthreads();

  // Write results back to global memory
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

  mmaOBTSKernelSparse_tiled_large<<<grid, block>>>(
      bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, metadata, sparseMatrixA, B, C, M,
      N, K, nonzeroBlocks, blockInfo, relativeBlockIndexMapping, tileInfo);
}