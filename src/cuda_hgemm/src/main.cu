#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define BLOCK 4

#define HGEMM_FUNC(name)                                                       \
  void name(half *A, half *B, half *C, size_t M, size_t N, size_t K)
#define HGEMM_FUNC_SPARSE(name)                                                \
  void name(half *bcsrValuesA, half *B, half *C, size_t M, size_t N, size_t K, \
            size_t nonzeroBlocks, int *blockInfo,                              \
            int *relativeBlockIndexMapping)
#define HGEMM_FUNC_SPARSE24(name)                                              \
  void name(half *bcsrValuesA, char *metadata, half *sparseMatrixA, half *B,   \
            half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks,       \
            int *blockInfo, int *relativeBlockIndexMapping)
#define HGEMM_FUNC_SPARSE24_2(name)                                            \
  void name(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA,             \
            char *metadata, half *sparseMatrixA, half *B, half *C, size_t M,   \
            size_t N, size_t K, size_t nonzeroBlocks, int *blockInfo,          \
            int *relativeBlockIndexMapping)
#define HGEMM_FUNC_SPARSE2(name)                                               \
  void name(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B,    \
            half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks,       \
            int *blockInfo, int *relativeBlockIndexMapping)
#define HGEMM_FUNC_SPARSE24_2_tiled(name)                                      \
  void name(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA,             \
            char *metadata, half *sparseMatrixA, half *B, half *C, size_t M,   \
            size_t N, size_t K, size_t nonzeroBlocks, int *blockInfo,          \
            int *relativeBlockIndexMapping, int *tileInfo)

#define HGEMM_FUNC_SPLIT(name)                                                 \
  void name(/* 2:4  */ half *bcsrValuesA_sparse, int *bcsrRowPtrA_sparse,      \
            int *bcsrColIdxA_sparse, /* dense*/ half *bcsrValuesA_dense,       \
            int *bcsrRowPtrA_dense, int *bcsrColIdxA_dense,                    \
            /* extra*/ char *metadata, half *sparseMatrixA, half *B, half *C,  \
            size_t M, size_t N, size_t K, size_t nonzeroBlocks,                \
            int *blockInfo, int *relativeBlockIndexMapping)

HGEMM_FUNC(cublasTensorOp);

HGEMM_FUNC_SPARSE(mmaNaiveKernel);
HGEMM_FUNC_SPARSE(mmaTKernel);
HGEMM_FUNC_SPARSE24(mmaSTKernel);
HGEMM_FUNC_SPARSE24(mmaSTKernel_large);

HGEMM_FUNC_SPARSE2(mmaBKernel);
HGEMM_FUNC_SPARSE2(mmaBTKernel);
HGEMM_FUNC_SPARSE2(mmaCBTKernel);
HGEMM_FUNC_SPARSE2(mmaOBTKernel);
HGEMM_FUNC_SPARSE2(mmaOBTKernel_large);
HGEMM_FUNC_SPARSE2(mmaOBTKernel_tiled);
HGEMM_FUNC_SPARSE24_2(mmaOBTSKernel);
HGEMM_FUNC_SPARSE24_2(mmaOBTSKernel_large);
HGEMM_FUNC_SPARSE24_2_tiled(mmaOBTSKernel_tiled_large);
HGEMM_FUNC_SPLIT(mmaOBTSKernel_large_separate);

void preprocessing_mmaSTKernel(half *bcsrValuesA, char *metadata,
                               half *sparseMatrixA, size_t M, size_t N,
                               size_t K, size_t nonzeroBlocks, int *blockInfo,
                               int *relativeBlockIndexMapping);
void preprocessing_mmaSTKernel_large(half *bcsrValuesA, char *metadata,
                                     half *sparseMatrixA, size_t M, size_t N,
                                     size_t K, size_t nonzeroBlocks,
                                     int *blockInfo,
                                     int *relativeBlockIndexMapping);
void preprocessing_mmaOBTSKernel_tiled_large(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
    int *relativeBlockIndexMapping, int *tileInfo);

DEFINE_uint32(M, 16384, "M");
DEFINE_uint32(N, 64, "N");
DEFINE_uint32(K, 16384, "K");
// DEFINE_uint32(M, 121192, "M");
// DEFINE_uint32(N, 121192, "N");
// DEFINE_uint32(K, 121192, "K");
// DEFINE_uint32(M, 128, "M");
// DEFINE_uint32(N, 128, "N");
// DEFINE_uint32(K, 128, "K");
// DEFINE_uint32(M, 1024, "M");
// DEFINE_uint32(N, 1024, "N");
// DEFINE_uint32(K, 1024, "K");
// DEFINE_uint32(M, 1024, "M");
// DEFINE_uint32(N, 8, "N");
// DEFINE_uint32(K, 1024, "K");
// DEFINE_uint32(M, 2048, "M");
// DEFINE_uint32(N, 2048, "N");
// DEFINE_uint32(K, 2048, "K");
DEFINE_bool(enable_wmma, true, "test WMMA API");
DEFINE_bool(enable_mma, true, "test MMA PTX instruction");
DEFINE_uint32(warmup_iterations, 1,
              "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10,
              "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false,
            "check the GPU result against the cublas result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");
DEFINE_uint32(n_mult, 2, "n_mult * MMA_N = N");
// DEFINE_string(filename,
//               "./src/matrices/2_4_sparse_matrices/"
//               "2_4_sparse_mtx_1024_0.4000.mtx",
//               "input .mtx file");
// DEFINE_string(filename,
//               "./src/matrices/2_4_sparse_matrices/"
//               "2_4_sparse_mtx_128_0.5000.mtx",
//               "input .mtx file");
// DEFINE_string(filename,
//               "./src/matrices/2_4_sparse_matrices/"
//               "2_4_sparse_mtx_2048_0.1000.mtx",
//               "input .mtx file");
DEFINE_string(filename, "../sparse-gemm/build/mat_1d_1s_1024x1024_lg.mtx",
              "input .mtx file");
// DEFINE_string(filename,
// "../sparse-gemm/build/mat_50d_50s_128x128_sm.mtx",
//               "input .mtx file");
// DEFINE_string(filename,
//               "./src/matrices/band_matrices_2_4_sparse/"
//               "band_mtx_2_4_sparse_16384_32.mtx",
//               "input .mtx file");
// DEFINE_string(filename,
//               "./src/matrices/band_matrices_4_times/band_mtx_1024_512.mtx",
//               "input .mtx file");
// DEFINE_string(filename,
// "./src/matrices/suitesparse/cop20k_A/cop20k_A.mtx",
//               "input .mtx file");

void testBcsrBlocking() {
  SparseMatrix testMatrix(
      "TestMatrix",
      "./src/matrices/2_4_sparse_matrices/2_4_sparse_mtx_64_0.5000.mtx");

  // Get matrix dimensions and check they're valid
  size_t rows = testMatrix.getRow();
  size_t cols = testMatrix.getCol();
  std::cout << "Testing matrix of size " << rows << "x" << cols << std::endl;

  // Print original matrix pattern for visualization (up to 32x32)
  const int DISPLAY_SIZE = 64;
  size_t display_rows = std::min(rows, (size_t)DISPLAY_SIZE);
  size_t display_cols = std::min(cols, (size_t)DISPLAY_SIZE);

  std::cout << "\nOriginal matrix pattern (showing first " << display_rows
            << "x" << display_cols << "):\n";
  testMatrix.makeDenseArray();
  testMatrix.moveToHost();

  for (size_t i = 0; i < display_rows; i++) {
    for (size_t j = 0; j < display_cols; j++) {
      // float value = 0.0f;
      // for (int k = testMatrix.getBcsrRowPtrHost()[i / MMA_M];
      //      k < testMatrix.getBcsrRowPtrHost()[i / MMA_M + 1]; k++) {
      //   if (testMatrix.getBcsrColIdxHost()[k] <= j &&
      //       j < testMatrix.getBcsrColIdxHost()[k] + MMA_K) {
      //     float val = __half2float(
      //         testMatrix
      //             .getBcsrValuesHost()[k * MMA_M * MMA_K + (i % MMA_M) *
      //             MMA_K +
      //                                  (j % MMA_K)]);
      //     if (val != 0.0f) {
      //       value = val;
      //       break;
      //     }
      //   }
      // }
      float value = testMatrix.getHostPtr()[i * testMatrix.getCol() + j];
      printf("%2.0f ", value); // Print with 2 chars width, 0 decimal places
    }
    std::cout << "\n";
  }

  // Count original nonzeros and store their positions
  size_t originalNonZeros = 0;
  struct NonZeroValue {
    size_t row;
    size_t col;
    float value;
  };
  std::vector<NonZeroValue> originalValues;

  for (size_t i = 0; i < rows; i++) {
    for (size_t block = testMatrix.getBcsrRowPtrHost()[i / MMA_M];
         block < testMatrix.getBcsrRowPtrHost()[i / MMA_M + 1]; block++) {
      int blockCol = testMatrix.getBcsrColIdxHost()[block];
      for (size_t subRow = 0;
           subRow < MMA_M && (i / MMA_M * MMA_M + subRow) < rows; subRow++) {
        for (size_t subCol = 0; subCol < MMA_K && (blockCol + subCol) < cols;
             subCol++) {
          float val = __half2float(
              testMatrix.getBcsrValuesHost()[block * MMA_M * MMA_K +
                                             subRow * MMA_K + subCol]);
          if (val != 0.0f) {
            originalNonZeros++;
            originalValues.push_back({i / MMA_M * MMA_M + (i % MMA_M) + subRow,
                                      (size_t)blockCol + subCol, val});
          }
        }
      }
    }
  }
  std::cout << "\nOriginal nonzero elements: " << originalNonZeros << std::endl;

  // Try blocking with size 2
  // int blockSize = 2;
  std::cout << "\nTesting blocking" << std::endl;
  testMatrix.bcsrBlocking();

  // Print merged blocks and count nonzeros
  std::cout << "\nNumber of merged nonzero blocks: "
            << testMatrix.getMergedNonzeroBlocks() << "\n";
  size_t mergedNonZeros = 0;

  // Only print first few blocks to avoid overwhelming output
  const int MAX_BLOCKS_TO_PRINT = 40;
  size_t blocks_to_print = std::min(testMatrix.getMergedNonzeroBlocks(),
                                    (size_t)MAX_BLOCKS_TO_PRINT);

  // Print row pointers for debugging
  std::cout << "Row pointers: ";
  for (int i = 0; i <= testMatrix.getRow() / (BLOCK * MMA_M); i++) {
    std::cout << testMatrix.getMergedBcsrRowPtrHost()[i] << " ";
  }
  std::cout << "\n";

  std::cout << "\nOriginal matrix pattern (showing first " << display_rows
            << "x" << display_cols << "):\n";
  testMatrix.makeDenseArray();
  testMatrix.moveToHost();

  for (size_t i = 0; i < display_rows; i++) {
    for (size_t j = 0; j < display_cols; j++) {

      float value = testMatrix.getHostPtr()[i * testMatrix.getCol() + j];
      printf("%2.0f ", value); // Print with 2 chars width, 0 decimal places
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // print merged colidx and rowptr
  std::cout << "\n";
  for (size_t i = 0; i < testMatrix.getMergedNonzeroBlocks(); i++) {
    std::cout << "Merged ColIdx: " << testMatrix.getMergedBcsrColIdxHost()[i]
              << "\n";
  }
  for (size_t i = 0; i < testMatrix.getMergedNonzeroBlocks(); i++) {
    std::cout << "Merged RowPtr: " << testMatrix.getMergedBcsrRowPtrHost()[i]
              << "\n";
  }

  for (size_t i = 0; i < blocks_to_print; i++) {
    std::cout << "\nMerged Block " << i << " (starting at column "
              << testMatrix.getMergedBcsrColIdxHost()[i] << "):\n";
    for (int row = 0; row < BLOCK * MMA_M; row++) {
      for (int col = 0; col < BLOCK * MMA_K; col++) {
        float val = __half2float(
            testMatrix.getMergedBcsrValuesHost()
                [i * BLOCK * BLOCK * MMA_M * MMA_K +
                 +(row / MMA_M) * BLOCK * MMA_K * MMA_M +
                 (col / (MMA_M)) * MMA_M * MMA_K + (col % (MMA_K)) +
                 ((row % MMA_M) * MMA_K)]); // which small block

        // float val =
        //     __half2float(testMatrix.getMergedBcsrValuesHost()
        //                      [i * blockSize * blockSize * MMA_M * MMA_K +
        //                       row * blockSize * MMA_K * MMA_M + col]);

        //  row * blockSize * MMA_K + col]);
        printf("%4.1f ", val);
        if (val != 0.0f) {
          mergedNonZeros++;
        }
      }
      std::cout << "\n";
    }
  }

  size_t numColRegions =
      (testMatrix.getCol() + (MMA_K * BLOCK) - 1) / (MMA_K * BLOCK);
  size_t numRowRegions =
      (testMatrix.getRow() + (MMA_M * BLOCK) - 1) / (MMA_M * BLOCK);

  // Get pointers to the arrays
  int *blockInfo = testMatrix.getMergedBlockInfo_host();
  int *relativeMapping = testMatrix.getMergedRelativeBlockIndexMapping_host();

  // Print mergedBlockInfo
  std::cout << "Merged Block Info:" << std::endl;
  for (size_t i = 0; i < numRowRegions; i++) {
    for (size_t j = 0; j < numColRegions; j++) {
      std::cout << blockInfo[i * numColRegions + j] << " ";
    }
    std::cout << std::endl;
  }

  // Print mergedRelativeBlockIndexMapping
  std::cout << "\nMerged Relative Block Index Mapping:" << std::endl;
  for (size_t i = 0; i < numRowRegions; i++) {
    for (size_t j = 0; j < numColRegions; j++) {
      std::cout << relativeMapping[i * numColRegions + j] << " ";
    }
    std::cout << std::endl;
  }

  // Continue counting nonzeros for remaining blocks
  for (size_t i = blocks_to_print; i < testMatrix.getMergedNonzeroBlocks();
       i++) {
    for (int j = 0; j < BLOCK * BLOCK * MMA_M * MMA_K; j++) {
      if (__half2float(testMatrix.getMergedBcsrValuesHost()
                           [i * BLOCK * BLOCK * MMA_M * MMA_K + j]) != 0.0f) {
        mergedNonZeros++;
      }
    }
  }

  std::cout << "\nMerged nonzero elements: " << mergedNonZeros << std::endl;
  std::cout << "Storage efficiency: "
            << (float)originalNonZeros / mergedNonZeros * 100 << "% ("
            << originalNonZeros << "/" << mergedNonZeros << ")\n";

  // Verify that all original non-zero values are present in merged blocks
  bool allValuesFound = true;
  for (const auto &originalValue : originalValues) {
    bool found = false;
    size_t row = originalValue.row;
    size_t col = originalValue.col;
    float val = originalValue.value;

    //     std::cout << "\nLooking for value " << val << " at (" << row <<
    //     ","
    //     << col
    //               << ")\n";

    // Search in merged blocks
    for (size_t mergedBlock = 0;
         mergedBlock < testMatrix.getMergedNonzeroBlocks() && !found;
         mergedBlock++) {

      int blockStartCol = testMatrix.getMergedBcsrColIdxHost()[mergedBlock];

      // Find which row contains this block by searching through row
      // pointers
      int rowIdx = 0;
      while (rowIdx < testMatrix.getRow() / (BLOCK * MMA_M) &&
             testMatrix.getMergedBcsrRowPtrHost()[rowIdx + 1] <= mergedBlock) {
        rowIdx++;
      }
      int blockStartRow = rowIdx * BLOCK * MMA_M;

      //  std::cout << "  Checking block " << mergedBlock << " starting at
      //  ("
      //            << blockStartRow << "," << blockStartCol << ")\n";

      // Check if this block could contain our value
      if (blockStartCol <= col && col < blockStartCol + BLOCK * MMA_K &&
          blockStartRow <= row && row < blockStartRow + BLOCK * MMA_M) {

        // Calculate position within merged block
        size_t relRow = row - blockStartRow;
        size_t relCol = col - blockStartCol;

        size_t offset =
            mergedBlock * BLOCK * BLOCK * MMA_M * MMA_K +
            (relRow / MMA_M) * (BLOCK * MMA_K * MMA_M) + // block row offset
            (relRow % MMA_M) * MMA_K +           // within block row offset
            (relCol / MMA_K) * (MMA_K * MMA_M) + // block col offset
            (relCol % MMA_K);

        float mergedVal =
            __half2float(testMatrix.getMergedBcsrValuesHost()[offset]);

        //    std::cout << "    Found value " << mergedVal
        //              << " at relative position (" << relRow << "," <<
        //              relCol
        //              << ") offset " << offset << "\n";

        if (std::abs(mergedVal - val) < 1e-5) {
          found = true;
          break;
        }
      }
    }

    if (!found) {
      //  std::cout << "Value " << val << " at (" << row << "," << col
      //            << ") not found in merged blocks!\n";
      allValuesFound = false;
    }
  }

  if (allValuesFound) {
    std::cout << "All values successfully found in merged blocks!\n";
  }
}
// DEFINE_string(filename, "./src/matrices/suitesparse/mip1/mip1.mtx",
//               "input .mtx file");

int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  omp_set_num_threads(FLAGS_cpu_procs);
  HGEMM_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

  cudaDeviceProp dev_prop;
  HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
  HLOG("CUDA HGEMM start with %u CPU processes on the %u-th GPU: %s",
       FLAGS_cpu_procs, FLAGS_gpu_rank, dev_prop.name);

  int driver_version = 0;
  int runtime_version = 0;
  HGEMM_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
  HGEMM_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
  HLOG("CUDA driver version / runtime version: %d.%d / %d.%d",
       driver_version / 1000, (driver_version % 100) / 10,
       runtime_version / 1000, (runtime_version % 100) / 10);
  HLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major,
       dev_prop.minor);
  HLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores",
       dev_prop.multiProcessorCount,
       convert_SM_to_cores(dev_prop.major, dev_prop.minor),
       convert_SM_to_cores(dev_prop.major, dev_prop.minor) *
           dev_prop.multiProcessorCount);
  HLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)",
       static_cast<double>(dev_prop.clockRate) * 1e-3,
       static_cast<double>(dev_prop.clockRate) * 1e-6);
  HLOG("Memory clock rate: %.0f MHz (%0.2f GHz)",
       static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
       static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
  HLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
  HLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
       static_cast<double>(dev_prop.totalGlobalMem) / 1048576,
       dev_prop.totalGlobalMem);
  HLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)",
       static_cast<double>(dev_prop.totalConstMem) / 1024,
       dev_prop.totalConstMem);
  HLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
       static_cast<double>(dev_prop.sharedMemPerBlock) / 1024,
       dev_prop.sharedMemPerBlock);
  HLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
       static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024,
       dev_prop.sharedMemPerMultiprocessor);
  HLOG("L2 cache size: %.0f KBytes (%d Bytes)",
       static_cast<double>(dev_prop.l2CacheSize) / 1024, dev_prop.l2CacheSize);
  HLOG("Total number of registers available per block: %d",
       dev_prop.regsPerBlock);
  HLOG("Warp size: %d", dev_prop.warpSize);
  HLOG("Max number of threads per multiprocessor: %d",
       dev_prop.maxThreadsPerMultiProcessor);
  HLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
  HLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
       dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1],
       dev_prop.maxThreadsDim[2]);
  HLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)",
       dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
       dev_prop.maxGridSize[2]);

  HLOG("A (%u x %u) * B (%u x %u) = C (%u x %u), N_MULT: %u", FLAGS_M, FLAGS_K,
       FLAGS_K, FLAGS_N, FLAGS_M, FLAGS_N, FLAGS_n_mult);
  HLOG("Profiling: enable wmma: %d, enable mma: %d, warmup iterations: %u, "
       "profiling iterations: %u, sleep duration: "
       "%u ms, enable check: %d",
       FLAGS_enable_wmma, FLAGS_enable_mma, FLAGS_warmup_iterations,
       FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);

  std::string file(FLAGS_filename);
  HLOG("Input .mtx: %s", file.data());
  Tester tester(FLAGS_M, FLAGS_N, FLAGS_K, FLAGS_warmup_iterations,
                FLAGS_profiling_iterations, FLAGS_sleep_duration,
                FLAGS_enable_check, FLAGS_n_mult, file.data(), true);

  // auto result_mmaT = tester.evaluateSparse(mmaTKernel, "Mma-T-Kernel");
  // auto result_cublas = tester.evaluate(cublasTensorOp, "Cublas-Tensor-Op");

  // auto result_mmaST = tester.evaluateSparse24(
  //     mmaSTKernel, preprocessing_mmaSTKernel, "Mma-ST-Kernel");
  // auto result_mmaST_large = tester.evaluateSparse24(
  //     mmaSTKernel_large, preprocessing_mmaSTKernel_large,
  //     "Mma-ST-Kernel-large", true);

  // auto result_mmaBT = tester.evaluateSparse2(mmaBTKernel, "Mma-BT-Kernel");
  // auto result_mmaCBT = tester.evaluateSparse2(mmaCBTKernel,
  // "Mma-CBT-Kernel"); auto result_mmaOBT =
  // tester.evaluateSparse2(mmaOBTKernel, "Mma-OBT-Kernel"); auto
  // result_mmaOBT_large =
  //     tester.evaluateSparse2(mmaOBTKernel_large, "Mma-OBT-large-Kernel");
  // auto result_mmaOBT_tiled =
  //     tester.evaluateSparse2_tiled(mmaOBTKernel_tiled,
  //     "Mma-OBT-Kernel-tiled");

  // auto result_mmaOBTS = tester.evaluateSparse24_2(
  //     mmaOBTSKernel, preprocessing_mmaSTKernel, "Mma-OBTS-Kernel");
  auto result_mmaOBTS_large = tester.evaluateSparse24_2(
      mmaOBTSKernel_large, preprocessing_mmaSTKernel_large,
      "Mma-OBTS-Kernel-large", true);
  auto result_mmaOBTS_large_separate = tester.evaluateSplit(
      mmaOBTSKernel_large_separate,    //  the host stub with dense+sparse
      preprocessing_mmaSTKernel_large, //  your 2:4 metadata packer
      "Mma-OBTS-large-separate", true);

  std::cout << "\nResults:\n";
  // std::cout << "Mma-T-Kernel: " << result_mmaT.first << " ms, "
  //           << result_mmaT.second << " throughput\n";
  // std::cout << "Cublas-Tensor-Op: " << result_cublas.first << " ms, "
  //           << result_cublas.second << " throughput\n";
  // std::cout << "Mma-ST-Kernel: " << result_mmaST.first << " ms, "
  //           << result_mmaST.second << " throughput\n";
  // std::cout << "Mma-ST-Kernel-large: " << result_mmaST_large.first << " ms, "
  //           << result_mmaST_large.second << " throughput\n";
  // std::cout << "Mma-BT-Kernel: " << result_mmaBT.first << " ms, "
  //           << result_mmaBT.second << " throughput\n";
  // std::cout << "Mma-CBT-Kernel: " << result_mmaCBT.first << " ms, "
  //           << result_mmaCBT.second << " throughput\n";
  // std::cout << "Mma-OBT-Kernel: " << result_mmaOBT.first << " ms, "
  //           << result_mmaOBT.second << " throughput\n";
  // std::cout << "Mma-OBT-large-Kernel: " << result_mmaOBT_large.first << "ms,"
  //           << result_mmaOBT_large.second << " throughput\n";
  // std::cout << "Mma-OBT-Kernel-tiled: " << result_mmaOBT_tiled.first << "ms,"
  //           << result_mmaOBT_tiled.second << " throughput\n";
  // std::cout << "Mma-OBTS-Kernel: " << result_mmaOBTS.first << " ms, "
  //           << result_mmaOBTS.second << " throughput\n";
  std::cout << "Mma-OBTS-Kernel-large: " << result_mmaOBTS_large.first
            << " ms, " << result_mmaOBTS_large.second << " throughput\n";
  std::cout << "Mma-OBTS-Kernel-large-separate: "
            << result_mmaOBTS_large_separate.first << " ms, "
            << result_mmaOBTS_large_separate.second << " throughput\n";
  // tester.evaluateSparse24_2_tiled(mmaOBTSKernel_tiled_large,
  //                                 preprocessing_mmaOBTSKernel_tiled_large,
  //                                 "Mma-OBTS-Kernel-tiled-large", true);
  // still need to work on this

  GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

  HLOG("Done");

  return 0;
}
