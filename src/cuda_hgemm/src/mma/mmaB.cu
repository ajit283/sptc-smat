#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__global__ void mmaBKernelSparse(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks, int* blockInfo, int* relativeBlockIndexMapping) {
    //mmaBKernel
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    size_t blockRow = blockIdx.y;
    size_t blockCol = blockIdx.x;

    size_t colRegions = (K + MMA_K - 1) / (MMA_K);

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    float RC[2] = {0, 0};
#pragma unroll
    for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1]; ptr++) {
        size_t i = bcsrColIdxA[ptr] / MMA_K;
        size_t blockIndex = blockRow * colRegions + i;

        size_t relativeIndex = relativeBlockIndexMapping[blockIndex];

        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&bcsrValuesA[(relativeIndex) * MMA_M * MMA_K + (lane_id / 2) * MMA_K]) + lane_id % 2);
        
        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }
        
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        half a = __float2half(0);
        for (int rowi = 0; rowi < 16; rowi++) {
            *(&C[(warp_row + lane_id) * N + warp_col] + rowi) =  *(&C[(warp_row + lane_id) * N + warp_col] + rowi) + *((&bcsrValuesA[(relativeIndex) * MMA_M * MMA_K + (lane_id / 2) * MMA_K] + rowi) + lane_id % 2) * *((&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2 + rowi) ;

        }

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

void mmaBKernel(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks, int* blockInfo, int* relativeBlockIndexMapping) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    mmaBKernelSparse<<<grid, block>>>(bcsrValuesA, bcsrRowPtrA, bcsrColIdxA, B, C, M, N, K, nonzeroBlocks, blockInfo, relativeBlockIndexMapping);

}
