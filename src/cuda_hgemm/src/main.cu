#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#include <iomanip>
#include <utility>
#include <vector>
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
void preprocessing_mmaOBTSKernel_large_separate(
    half *bcsrValuesA, char *metadata, half *sparseMatrixA, size_t M, size_t N,
    size_t K, size_t nonzeroBlocks, int *blockInfo,
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
// DEFINE_string(filename, "../sparse-gemm/build/mat_5d_5s_1024x1024_lg.mtx",
//               "input .mtx file");
// DEFINE_string(filename, "../sparse-gemm/build/mat_1d_1s_1024x1024_lg.mtx",
//               "input .mtx file");
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

  std::vector<
      std::pair<std::pair<std::string, std::string>, std::tuple<int, int, int>>>
      filenames = {
          // {{"../sparse-gemm/build/mat_0d_1s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_0d_1s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},
          // {{"../sparse-gemm/build/mat_1d_0s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_1d_0s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},
          // {{"../sparse-gemm/build/mat_1d_1s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_1d_1s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},
          // {{"../sparse-gemm/build/mat_5d_5s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_5d_5s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},

          // {{"../sparse-gemm/build/mat_0d_10s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_0d_10s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},
          // {{"../sparse-gemm/build/mat_10d_0s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_10d_0s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},

          // {{"../sparse-gemm/build/mat_10d_10s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_10d_10s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},

          // {{"../sparse-gemm/build/mat_0d_20s_1024x1024_sm.mtx",
          //   "../sparse-gemm/build/mat_0d_20s_1024x1024_lg.mtx"},
          //  {16384, FLAGS_n_mult * MMA_N, 16384}},

          {{"../sparse-gemm/build/mat_20d_0s_1024x1024_sm.mtx",
            "../sparse-gemm/build/mat_20d_0s_1024x1024_lg.mtx"},
           {16384, FLAGS_n_mult * MMA_N, 16384}},
          // {{"./src/matrices/2_4_sparse_matrices/2_4_sparse_mtx_2048_0.1000.mtx",
          //   "./src/matrices/2_4_sparse_matrices/"
          //   "2_4_sparse_mtx_2048_0.1000.mtx"},
          //  {2048, 2048, 2048}},
      };

  std::vector<std::tuple<std::string,
                         std::vector<std::tuple<std::string, float, float>>>>
      all_results;

  for (auto &filename : filenames) {

    std::string file = filename.first.first;
    std::string file_large = filename.first.second;

    //     print filnames
    std::cout << file << std::endl;
    std::cout << file_large << std::endl;

    std::string matrix_name = file.substr(file.find_last_of("/") + 1);

    HLOG("Input .mtx: %s", file.data());
    Tester tester(std::get<0>(filename.second), std::get<1>(filename.second),
                  std::get<2>(filename.second), FLAGS_warmup_iterations,
                  FLAGS_profiling_iterations, FLAGS_sleep_duration,
                  FLAGS_enable_check, FLAGS_n_mult, file, true);
    Tester tester_large(std::get<0>(filename.second),
                        std::get<1>(filename.second),
                        std::get<2>(filename.second), FLAGS_warmup_iterations,
                        FLAGS_profiling_iterations, FLAGS_sleep_duration,
                        FLAGS_enable_check, FLAGS_n_mult, file_large, true);

    // Create a vector for this matrix's results
    std::vector<std::tuple<std::string, float, float>> matrix_results;

    // Store results in variables to avoid running each kernel twice
    auto result_mmaT = tester.evaluateSparse(mmaTKernel, "Mma-T-Kernel");
    // auto result_cublas = tester.evaluate(cublasTensorOp, "Cublas-Tensor-Op");
    auto result_mmaST = tester.evaluateSparse24(
        mmaSTKernel, preprocessing_mmaSTKernel, "Mma-ST-Kernel");
    auto result_mmaST_large = tester.evaluateSparse24(
        mmaSTKernel_large, preprocessing_mmaSTKernel_large,
        "Mma-ST-Kernel-large", true);
    auto result_mmaBT = tester.evaluateSparse2(mmaBTKernel, "Mma-BT-Kernel");
    auto result_mmaCBT = tester.evaluateSparse2(mmaCBTKernel, "Mma-CBT-Kernel");
    auto result_mmaOBT = tester.evaluateSparse2(mmaOBTKernel, "Mma-OBT-Kernel");
    auto result_mmaOBT_large =
        tester.evaluateSparse2(mmaOBTKernel_large, "Mma-OBT-large-Kernel");
    auto result_mmaOBT_tiled = tester.evaluateSparse2_tiled(
        mmaOBTKernel_tiled, "Mma-OBT-Kernel-tiled");
    auto result_mmaOBTS = tester.evaluateSparse24_2(
        mmaOBTSKernel, preprocessing_mmaSTKernel, "Mma-OBTS-Kernel");
    auto result_mmaOBTS_large = tester_large.evaluateSparse24_2(
        mmaOBTSKernel_large, preprocessing_mmaSTKernel_large,
        "Mma-OBTS-Kernel-large", true);
    auto result_mmaOBTS_large_separate = tester_large.evaluateSplit(
        mmaOBTSKernel_large_separate, preprocessing_mmaSTKernel_large,
        "Mma-OBTS-large-separate", true);

    // Add each result to the current matrix's results vector
    matrix_results.push_back(
        std::make_tuple("Mma-T-Kernel", result_mmaT.first, result_mmaT.second));
    //     matrix_results.push_back(std::make_tuple(
    //         "Cublas-Tensor-Op", result_cublas.first, result_cublas.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-BT-Kernel", result_mmaBT.first, result_mmaBT.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-CBT-Kernel", result_mmaCBT.first, result_mmaCBT.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-ST-Kernel", result_mmaST.first, result_mmaST.second));
    matrix_results.push_back(std::make_tuple("Mma-ST-Kernel-large",
                                             result_mmaST_large.first,
                                             result_mmaST_large.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-OBT-Kernel", result_mmaOBT.first, result_mmaOBT.second));
    matrix_results.push_back(std::make_tuple("Mma-OBT-large-Kernel",
                                             result_mmaOBT_large.first,
                                             result_mmaOBT_large.second));
    matrix_results.push_back(std::make_tuple("Mma-OBT-Kernel-tiled",
                                             result_mmaOBT_tiled.first,
                                             result_mmaOBT_tiled.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-OBTS-Kernel", result_mmaOBTS.first, result_mmaOBTS.second));
    matrix_results.push_back(std::make_tuple("Mma-OBTS-Kernel-large",
                                             result_mmaOBTS_large.first,
                                             result_mmaOBTS_large.second));
    matrix_results.push_back(std::make_tuple(
        "Mma-OBTS-large-separate", result_mmaOBTS_large_separate.first,
        result_mmaOBTS_large_separate.second));

    // Add this matrix's name and results to the all_results vector
    all_results.push_back(std::make_tuple(matrix_name, matrix_results));
  }

  // Print results for each matrix
  for (const auto &matrix_entry : all_results) {
    std::string matrix_name = std::get<0>(matrix_entry);
    const auto &matrix_results = std::get<1>(matrix_entry);

    std::cout << "\nResults for matrix: " << matrix_name << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto &result : matrix_results) {
      std::cout << std::get<0>(result) << ": " << std::get<1>(result) << " ms, "
                << std::get<2>(result) << " throughput\n";
    }
    std::cout << "\n";
  }

  // Optional: Print comparative summary table
  std::cout << "\nComparative Summary\n";
  std::cout << std::string(80, '=') << "\n";
  std::cout << std::left << std::setw(25) << "Kernel";

  // Print matrix names as headers
  for (const auto &matrix_entry : all_results) {
    std::cout << std::left << std::setw(25) << std::get<0>(matrix_entry);
  }
  std::cout << "\n" << std::string(80, '-') << "\n";

  // Get the first matrix's results to know which kernels to iterate through
  if (!all_results.empty()) {
    const auto &first_matrix_results = std::get<1>(all_results[0]);

    // For each kernel type
    for (size_t i = 0; i < first_matrix_results.size(); i++) {
      std::cout << std::left << std::setw(25)
                << std::get<0>(first_matrix_results[i]);

      // Print each matrix's result for this kernel
      for (const auto &matrix_entry : all_results) {
        const auto &matrix_results = std::get<1>(matrix_entry);
        const auto &kernel_result = matrix_results[i];
        std::cout << std::left << std::setw(25)
                  << (std::to_string(std::get<1>(kernel_result)) + " ms");
      }
      std::cout << "\n";
    }
  }
  GFLAGS_NAMESPACE::ShutDownCommandLineFlags();
  HLOG("Done");
  return 0;
}