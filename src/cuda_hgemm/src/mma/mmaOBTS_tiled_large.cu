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

//---------------------------------------------------------------------
// Preprocessing kernel with tiling for K=32 sparse blocks.
// This kernel compresses a dense 16×32 block into a sparse format using a 2:4
// scheme, where each row is reduced from 32 half elements to 16 half elements.
// The compressed data is stored in sparseMatrixA (using 2 int4’s per row)
// and metadata (2 bytes per row) is stored in metadata.
__global__ void preprocessing_mmaSTKernelSparse_large_tiled(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping) {

  // Compute the tile indices for this CUDA block.
  // (Each grid block covers a tile of size (MMA_M*BLOCK) x (MMA_N*BLOCK))
  const int tile_row = blockIdx.y; // tile index in rows
  const int tile_col = blockIdx.x; // tile index in columns

  // Each CUDA block contains BLOCK*BLOCK warps.
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id_y = warp_id / BLOCK; // sub-tile row index [0, BLOCK-1]
  const int warp_id_x = warp_id % BLOCK; // sub-tile col index [0, BLOCK-1]

  // Compute the global (dense) block indices.
  // For tiling, each sub-tile corresponds to one "block" of size MMA_M×MMA_K.
  // The total number of block columns across the K dimension is:
  int colRegions = div_ceil(K, MMA_K * BLOCK);
  int globalBlockRow = tile_row * BLOCK + warp_id_y; // overall block row index
  int globalBlockCol = tile_col * BLOCK + warp_id_x; // overall block col index
  int globalBlockIndex = globalBlockRow * colRegions + globalBlockCol;

  // Compute the origin (in the dense A matrix) for this block.
  size_t block_origin_row = globalBlockRow * MMA_M;
  size_t block_origin_col = globalBlockCol * MMA_N;
  if (block_origin_row >= M || block_origin_col >= N)
    return;

  // Get sparsity info. (0 means empty block; 1 indicates a sparse block.)
  int sparsityInfo = blockInfo[globalBlockIndex];
  if (sparsityInfo == 0)
    return;
  int relativeIndex = relativeBlockIndexMapping[globalBlockIndex];

  // Process only sparse blocks (sparsityInfo == 1). (Dense blocks could be
  // handled similarly.)
  if (sparsityInfo == 1) {
    // Shared memory buffers for temporary storage.
    // A_smem holds the dense block row data that will be compressed.
    // Meta_smem holds metadata bytes for each row.
    __shared__ half A_smem[MMA_M][MMA_K / 2]; // Each row will have MMA_K/2 = 16
                                              // half's after 2:4 compression.
    __shared__ char Meta_smem[MMA_M]
                             [MMA_K / 8]; // For 32 elements per row, 32/8 = 4
                                          // bytes are available per row.

    // We assume two lanes cooperate to process one row in the block.
    int row_in_block = lane_id / 2; // row index within the block (0 to MMA_M-1)
    int sub_lane =
        lane_id % 2; // which half of the row this lane is responsible for

    // Compute source pointer for the dense block from bcsrValuesA.
    // The dense block is stored contiguously at offset:
    //    relativeIndex * (MMA_M * MMA_K)
    // Each row has MMA_K (32) half elements.
    half *src = (half *)(((long4 *)&bcsrValuesA[relativeIndex * MMA_M * MMA_K +
                                                row_in_block * MMA_K]) +
                         sub_lane);

    // Temporary storage for the compressed row segment.
    // After compression each row (of 32 elements) becomes 16 half elements.
    // We assume that each lane (of the two handling the row) produces 8 half
    // elements.
    half src_sparse[8];
    // Two metadata bytes per row (one per 8-element half).
    char cur_meta[2];
    cur_meta[0] = 0;
    cur_meta[1] = 0;

// Loop over two parts. Each part processes 8 original half elements.
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      // Each part is divided into 2 groups of 4 elements.
      for (int j = 0; j < 2; ++j) {
        int cur_src_sparse = 0;
        int base_idx = part * 8 + j * 4; // starting index in this part
        for (int i = 0; i < 4; ++i) {
          half val = src[base_idx + i];
          if (val != __float2half(0.0f)) {
            // Store the nonzero half element.
            src_sparse[j * 4 + cur_src_sparse] = val;
            // Pack metadata bits: record the position (shifted appropriately).
            // (This shifting scheme is illustrative; adjust as needed for your
            // encoding.)
            cur_meta[part] |= (i << (6 - (2 * (1 - cur_src_sparse) + (4 * j))));
            cur_src_sparse++;
          }
        }
      }
    }

    // Write out metadata and compressed data.
    // For the compressed data:
    // - Each sparse block tile is stored in sparseMatrixA as 16 rows.
    // - Each row has 16 half elements, which are stored as 2 int4’s (each int4
    // packs 8 half elements).
    int dataOffset =
        relativeIndex * MMA_M * 2; // 2 int4’s per row for MMA_M rows.
    // For metadata:
    // - We store 2 bytes per row; assume a contiguous array.
    int metaOffset = relativeIndex * MMA_M * 2;
    // Write metadata for this row.
    metadata[metaOffset + row_in_block * 2 + sub_lane] = cur_meta[sub_lane];
    // Write the compressed row segment.
    // Each row’s compressed data is split between two lanes.
    // Here we reinterpret src_sparse (8 half elements) as an int4.
    // *((int4 *)(sparseMatrixA + dataOffset + row_in_block * 2 + sub_lane)) =
    //     *((int4 *)src_sparse);
    int dataIndex = relativeIndex * (MMA_M * 2) + row_in_block * 2 + sub_lane;
    ((int4 *)sparseMatrixA)[dataIndex] = *((int4 *)src_sparse);
  }
  // (Optionally, handle sparsityInfo == 2 for dense blocks.)
}

__global__ void mmaOBTSKernelSparse_tiled_large(
    half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C,
    size_t M, size_t N, size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping, int *tileInfo) {

  const size_t warp_row = blockIdx.y * MMA_M * BLOCK;
  const size_t warp_col = blockIdx.x * MMA_N * BLOCK;

  const size_t warp_id = threadIdx.x / WARP_SIZE;

  const size_t warp_id_y = (threadIdx.x / WARP_SIZE) / BLOCK;
  const size_t warp_id_x = (threadIdx.x / WARP_SIZE) % BLOCK;

  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  size_t colRegions = (K + MMA_K * BLOCK - 1) / (MMA_K * BLOCK);

  if (warp_row + warp_id_y * MMA_M >= M || warp_col + warp_id_x * MMA_N >= N) {
    return;
  }

  __shared__ half
      A_smem[NUM_STAGES][BLOCK][BLOCK][MMA_M][(MMA_K + ALIGNMENT_OFFSET)];
  __shared__ half
      B_smem[NUM_STAGES][BLOCK][BLOCK][MMA_N][(MMA_K + ALIGNMENT_OFFSET)];
  __shared__ half C_smem[BLOCK][BLOCK][MMA_M][MMA_N];

  const size_t lane_id = threadIdx.x % WARP_SIZE;

  uint32_t RA[NUM_STAGES][4];
  uint32_t RB[NUM_STAGES][2];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  // Load all pipeline stages.
  for (int stage = 0; stage < NUM_STAGES; ++stage) {
    pipe.producer_acquire();

    size_t ptr = bcsrRowPtrA[blockRow] + stage;
    if (ptr < bcsrRowPtrA[blockRow + 1]) {
      size_t i = bcsrColIdxA[ptr] / (MMA_K * BLOCK);
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      cuda::memcpy_async(((long4 *)(&A_smem[stage][warp_id_y][warp_id_x][(
                              lane_id / 2)][(ALIGNMENT_OFFSET)]) +
                          lane_id % 2),
                         (((long4 *)(&bcsrValuesA[(relativeIndex)*MMA_M *
                                                      MMA_K * BLOCK * BLOCK +
                                                  warp_id * MMA_M * MMA_K +
                                                  (lane_id / 2) * MMA_K]) +
                           lane_id % 2)),
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

      pipe.producer_commit();
    }
  }

  uint32_t RC[2] = {0, 0};
  int stage = 0;

#pragma unroll
  for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1];
       ptr++) {

    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
    __syncthreads();

    // moving along the main axis
    for (int i = 0; i < BLOCK; i++) {

      // uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
      //     &A_smem[stage][warp_id_y][i][lane_id / 2][(lane_id % 2) * 8]);
      uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
          &A_smem[stage][warp_id_y][i][(lane_id % 16)]
                 [(ALIGNMENT_OFFSET) + (lane_id / 16) * 8]);
      LDMATRIX_X4(RA[stage][0], RA[stage][1], RA[stage][2], RA[stage][3],
                  A_smem_lane_addr);

      // uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
      // &B_smem[stage][warp_id_y][i][(lane_id / 2) % 8]
      //        [(ALIGNMENT_OFFSET) + (lane_id % 2) * 8]);
      uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
          &B_smem[stage][warp_id_y][i][lane_id % 8]
                 [(ALIGNMENT_OFFSET) + ((lane_id / 8) % 2) * 8]);
      LDMATRIX_X2(RB[stage][0], RB[stage][1], B_smem_lane_addr);

      HMMA16816(RC[0], RC[1], RA[stage][0], RA[stage][1], RA[stage][2],
                RA[stage][3], RB[stage][0], RB[stage][1], RC[0], RC[1]);
    }

    __syncthreads();
    // Release the consumed stage.
    pipe.consumer_release();

    // Pre-load data for `num_stages` into the future.
    pipe.producer_acquire();

    size_t stage_ptr = ptr + NUM_STAGES;

    if (stage_ptr < bcsrRowPtrA[blockRow + 1]) {

      size_t i = bcsrColIdxA[stage_ptr] / (MMA_K * BLOCK);
      // skip empty block
      size_t blockIndex = blockRow * colRegions + i;

      size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

      size_t A_size = MMA_M * MMA_K * sizeof(half);
      size_t B_size = MMA_N * MMA_K * sizeof(half);

      cuda::memcpy_async(((int4 *)(&A_smem[stage][warp_id_y][warp_id_x]
                                          [(lane_id / 2)][(ALIGNMENT_OFFSET)]) +
                          lane_id % 2),
                         (((int4 *)(&bcsrValuesA[(relativeIndex)*MMA_M * MMA_K *
                                                     BLOCK * BLOCK +
                                                 warp_id * MMA_M * MMA_K +
                                                 (lane_id / 2) * MMA_K]) +
                           lane_id % 2)),
                         sizeof(int4), pipe);

      // For matrix B
      if (lane_id < MMA_N * 2) { // Original condition preserved
        cuda::memcpy_async(
            ((int4 *)(&B_smem[stage][warp_id_y][warp_id_x]
                             [MMA_N + (lane_id / 2)][(ALIGNMENT_OFFSET)]) +
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

void preprocessing_mmaOBTKernel_large(half *bcsrValuesA, char *metadata,
                                      half *sparseMatrixA, size_t M, size_t N,
                                      size_t K, size_t nonzeroBlocks,
                                      int *blockInfo,
                                      int *relativeBlockIndexMapping) {
  // Configure grid and block dimensions for tiling.
  // Each CUDA block covers a tile of (MMA_N*BLOCK) columns x (MMA_M*BLOCK)
  // rows.
  dim3 block(WARP_SIZE * BLOCK * BLOCK);
  dim3 grid(div_ceil(N, MMA_N * BLOCK), div_ceil(M, MMA_M * BLOCK));

  preprocessing_mmaSTKernelSparse_large_tiled<<<grid, block>>>(
      bcsrValuesA, metadata, sparseMatrixA, M, N, K, nonzeroBlocks, blockInfo,
      relativeBlockIndexMapping);
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
      bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, B, C, M, N, K, nonzeroBlocks,
      blockInfo, relativeBlockIndexMapping, tileInfo);
}