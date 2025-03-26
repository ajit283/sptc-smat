
#pragma once

#include "cuda_timer.h"
#include "matrix.h"
#include <cstring>
#include <memory>
#include <sys/time.h>
#include <time.h>
#include <utility>

class Tester {
public:
  explicit Tester(
      size_t M = 512, size_t N = 2048, size_t K = 1024,
      size_t warmup_iterations = 1, size_t profiling_iterations = 10,
      size_t sleep_duration = 100, bool enable_check = false, int N_mult = 1,
      std::string filename = "/scratch/czox/sparse_matrices/reordering_tests/"
                             "cop20k_A_reordered_transposed_reordered.mtx",
      bool enable_sparse_check = false)
      : m_M(M), m_N(N), m_K(K), m_warmup_iterations(warmup_iterations),
        m_profiling_iterations(profiling_iterations),
        m_sleep_duration(sleep_duration), m_enable_check(enable_check),
        m_N_mult(N_mult), m_file(filename),
        m_enable_sparse_check(enable_sparse_check) {
    HGEMM_CHECK_GT(m_M, 0);
    HGEMM_CHECK_GT(m_N, 0);
    HGEMM_CHECK_GT(m_K, 0);
    HGEMM_CHECK_GT(m_warmup_iterations, 0);
    HGEMM_CHECK_GT(m_profiling_iterations, 0);
    HGEMM_CHECK_GT(m_sleep_duration, 0);

    char *cstr = new char[filename.size() + 1];
    std::strcpy(cstr, filename.c_str());
    m_A_sparse = new SparseMatrix("Sparse Matrix A", cstr);
    m_A_sparse_large = new SparseMatrix("Sparse Matrix A", cstr, 32);
    // return;
    m_A = new Matrix(m_M, m_K, "Matrix A");
    HGEMM_CHECK(m_A);

    m_B = new Matrix(m_K, m_N, "Matrix B");
    HGEMM_CHECK(m_B);
    m_B_for_sparse = new Matrix(m_A_sparse->getCol(), m_N_mult * MMA_N,
                                "Matrix B for sparse");
    HGEMM_CHECK(m_B_for_sparse);

    m_C = new Matrix(m_M, m_N, "Matrix C");
    HGEMM_CHECK(m_C);
    m_C_for_sparse = new Matrix(m_A_sparse->getRow(), m_N_mult * MMA_N,
                                "Matrix C for sparse");
    HGEMM_CHECK(m_C_for_sparse);
    m_C_for_sparse->memSetHost();
    m_C_for_sparse->moveToDevice();
    m_C_for_sparse->memSetDevice();

    half *h_ptr = m_C_for_sparse->getHostPtr();
    size_t size = m_C_for_sparse->getRow() * m_C_for_sparse->getCol();
    for (size_t i = 0; i < size; i++) {
      if (__hisnan(h_ptr[i])) {
        printf("Found NaN in C at initialization: index %zu\n", i);
      }
    }

    m_base = new Matrix(m_M, m_N, "Matrix Base");
    HGEMM_CHECK(m_base);
    m_base_for_sparse =
        new Matrix(m_A_sparse->getRow(), m_N_mult * MMA_N, "Matrix Base");
    HGEMM_CHECK(m_base_for_sparse);
    m_base_for_sparse->memSetHost();
    m_base_for_sparse->moveToDevice();

    if (m_enable_check) {
      m_cuda_timer.start();
      cublas_tensor_op(m_A->getDevPtr(), m_B->getDevPtr(), m_base->getDevPtr(),
                       m_M, m_N, m_K);
      HLOG("Cublas-Tensor-Op use: %.3f ms", m_cuda_timer.end());
      m_base->moveToHost();
      m_base->memSetDevice();
    }

    printf("got here 1\n");

    if (m_enable_sparse_check) {
      printf("got here 4\n");
      m_cuda_timer.start();
      m_A_sparse->makeDenseArray();
      printf("got here 2\n");
      cublas_tensor_op(m_A_sparse->getDevPtr(), m_B_for_sparse->getDevPtr(),
                       m_base_for_sparse->getDevPtr(), m_A_sparse->getRow(),
                       m_base_for_sparse->getCol(), m_A_sparse->getCol());
      printf("got here 3\n");
      HLOG("Cublas-Tensor-Op use: %.3f ms", m_cuda_timer.end());
      m_base_for_sparse->moveToHost();
      m_base_for_sparse->memSetDevice();
    }
  }

  ~Tester() {
    if (m_A) {
      delete m_A;
      m_A = nullptr;
    }

    if (m_B) {
      delete m_B;
      m_B = nullptr;
    }

    if (m_C) {
      delete m_C;
      m_C = nullptr;
    }

    if (m_base) {
      delete m_base;
      m_base = nullptr;
    }

    if (m_A_sparse) {
      delete m_A_sparse;
      m_A_sparse = nullptr;
    }

    if (m_B_for_sparse) {
      delete m_B_for_sparse;
      m_B_for_sparse = nullptr;
    }

    if (m_C_for_sparse) {
      delete m_C_for_sparse;
      m_C_for_sparse = nullptr;
    }

    if (m_base_for_sparse) {
      delete m_base_for_sparse;
      m_base_for_sparse = nullptr;
    }
  }

  template <typename Func>
  void evaluate(Func &&hgemm, const std::string &name) {
    HLOG("----------------- Evaluating %s -----------------", name.c_str());
    usleep(m_sleep_duration * 1000);
    m_C->tearUp(m_base);

    // warm up
    m_cuda_timer.start();
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_M, m_N,
            m_K);
    }
    m_warmup_time = static_cast<double>(m_cuda_timer.end()) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_check && 0) {
      m_C->moveToHost();
      m_C->checkValue(m_base);
    }

    profile(std::forward<Func>(hgemm), name);
  }

  template <typename Func>
  void evaluateSparse(Func &&hgemm, const std::string &name) {
    HLOG("----------------- Sparse Evaluating %s -----------------",
         name.c_str());
    // HLOG("%d", m_A_sparse->getBlockInfo_host()[0]);
    usleep(m_sleep_duration * 1000);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    // warm up
    // m_cuda_timer.start();
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(m_A_sparse->getBcsrValues(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
    }
    m_warmup_time = 1; // static_cast<double>(m_cuda_timer.end()) /
                       // static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse(std::forward<Func>(hgemm), name);
  }

  template <typename Func>
  void evaluateSparse2(Func &&hgemm, const std::string &name) {
    HLOG("----------------- Sparse Evaluating %s -----------------",
         name.c_str());
    // HLOG("%d", m_A_sparse->getBlockInfo_host()[0]);
    usleep(m_sleep_duration * 1000);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    // warm up
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    // m_cuda_timer.start();
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(m_A_sparse->getBcsrValues(), m_A_sparse->getBcsrRowPtr(),
            m_A_sparse->getBcsrColIdx(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    m_warmup_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                     (t2.tv_usec - t1.tv_usec) / 1000.0) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse2(std::forward<Func>(hgemm), name);
  }
  template <typename Func>
  void evaluateSparse2_tiled(Func &&hgemm, const std::string &name) {
    HLOG("----------------- Sparse Evaluating %s -----------------",
         name.c_str());
    // HLOG("%d", m_A_sparse->getBlockInfo_host()[0]);
    usleep(m_sleep_duration * 1000);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    m_A_sparse->bcsrBlocking();

    std::cout << "N " << m_C_for_sparse->getCol() << " " << std::endl;
    // Detailed check of C matrix
    // Detailed check of C matrix
    size_t num_nans = 0;
    size_t M = m_C_for_sparse->getRow();
    size_t N = m_C_for_sparse->getCol();
    half *h_ptr = m_C_for_sparse->getHostPtr();

    std::cout << "Checking C matrix (" << M << " x " << N << ")..."
              << std::endl;
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        half val = h_ptr[i * N + j];
        float float_val = __half2float(val);
        if (isnan(float_val)) { // Using standard isnan on float instead
          std::cout << "NaN at position [" << i << "," << j << "]" << std::endl;
          num_nans++;
          if (num_nans <= 10) { // Only print first 10 occurrences
            std::cout << "NaN found at row " << i << " col " << j << std::endl;
          }
        }
      }
    }
    if (num_nans > 0) {
      std::cout << "Total NaN count: " << num_nans << std::endl;
    }

    std::cout << "C[0] " << __half2float(m_C_for_sparse->getHostPtr()[0]) << " "
              << __half2float(m_C_for_sparse->getHostPtr()[1]) << std::endl;
    m_C_for_sparse->memSetDevice();
    m_C_for_sparse->memSetHost();
    std::cout << "C[0] " << __half2float(m_C_for_sparse->getHostPtr()[0]) << " "
              << __half2float(m_C_for_sparse->getHostPtr()[1]) << std::endl;

    // warm up
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    // m_cuda_timer.start();
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(
          m_A_sparse->getMergedBcsrValues(), m_A_sparse->getMergedBcsrRowPtr(),
          m_A_sparse->getMergedBcsrColIdx(), m_B_for_sparse->getDevPtr(),
          m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
          m_C_for_sparse->getCol(), m_A_sparse->getCol(),
          m_A_sparse->getNonzeroblocks(), m_A_sparse->getMergedBlockInfo_dev(),
          m_A_sparse->getMergedRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    m_warmup_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                     (t2.tv_usec - t1.tv_usec) / 1000.0) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse2_tiled(std::forward<Func>(hgemm), name);
  }

  template <typename PreprocessFunc>
  std::pair<std::unique_ptr<Matrix>, std::unique_ptr<Matrix>>
  getPreprocessed(PreprocessFunc &&preprocess, bool large) {
    int k;
    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      k = 32;
      A_matrix = m_A_sparse_large;
    } else {
      k = 16;
    }
    size_t colRegions = (A_matrix->getCol() + k - 1) / (k);
    size_t rowRegions = (A_matrix->getRow() + MMA_M - 1) / (MMA_M);
    size_t nonzeroBlocks = A_matrix->getNonzeroblocks();
    HLOG("%d nonzero blocks", nonzeroBlocks);

    size_t metadata_size = nonzeroBlocks * MMA_M * (k / 8) / sizeof(half);
    auto metadata =
        std::make_unique<Matrix>(metadata_size, 1, "Matrix metadata");
    HGEMM_CHECK(metadata.get());
    metadata->memSetHost();
    metadata->moveToDevice();

    size_t sparseMatrixA_size =
        nonzeroBlocks * MMA_M * (k / 8) * sizeof(int2) / sizeof(half);
    auto sparseMatrixA =
        std::make_unique<Matrix>(sparseMatrixA_size, 1, "Sparse Matrix A");
    HGEMM_CHECK(sparseMatrixA.get());
    sparseMatrixA->memSetHost();

    sparseMatrixA->moveToDevice();

    HLOG("sparseMatrixA row: %d\n", sparseMatrixA->getRow());
    HLOG("sparseMatrixA col: %d\n", sparseMatrixA->getCol());
    HLOG("A_matrix row: %d \n", A_matrix->getRow());
    HLOG("A_matrix col: %d \n", A_matrix->getCol());

    usleep(m_sleep_duration * 1000);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    preprocess(A_matrix->getBcsrValues(), (char *)(metadata->getDevPtr()),
               sparseMatrixA->getDevPtr(), A_matrix->getRow(),
               A_matrix->getCol(), m_K, A_matrix->getNonzeroblocks(),
               A_matrix->getBlockInfo_dev(),
               A_matrix->getRelativeBlockIndexMapping_dev());

    return std::make_pair(std::move(metadata), std::move(sparseMatrixA));
  }
  template <typename PreprocessFunc>
  std::pair<std::unique_ptr<Matrix>, std::unique_ptr<Matrix>>
  getPreprocessedTile(PreprocessFunc &&preprocess, bool large) {
    int k;
    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      k = 32;
      A_matrix = m_A_sparse_large;
    } else {
      k = 16;
    }
    size_t colRegions = (A_matrix->getCol() + k - 1) / (k);
    size_t rowRegions = (A_matrix->getRow() + MMA_M - 1) / (MMA_M);
    size_t nonzeroBlocks = A_matrix->getNonzeroblocks();
    HLOG("%d nonzero blocks", nonzeroBlocks);

    size_t metadata_size =
        nonzeroBlocks * MMA_M * BLOCK * BLOCK * (k / 8) / sizeof(half);
    auto metadata =
        std::make_unique<Matrix>(metadata_size, 1, "Matrix metadata");
    HGEMM_CHECK(metadata.get());
    metadata->memSetHost();
    metadata->moveToDevice();

    size_t sparseMatrixA_size = nonzeroBlocks * MMA_M * (k / 8) * BLOCK *
                                BLOCK * sizeof(int2) / sizeof(half);
    auto sparseMatrixA =
        std::make_unique<Matrix>(sparseMatrixA_size, 1, "Sparse Matrix A");
    HGEMM_CHECK(sparseMatrixA.get());
    sparseMatrixA->memSetHost();

    sparseMatrixA->moveToDevice();

    HLOG("sparseMatrixA row: %d\n", sparseMatrixA->getRow());
    HLOG("sparseMatrixA col: %d\n", sparseMatrixA->getCol());
    HLOG("A_matrix row: %d \n", A_matrix->getRow());
    HLOG("A_matrix col: %d \n", A_matrix->getCol());

    usleep(m_sleep_duration * 1000);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    preprocess(A_matrix->getBcsrValues(), (char *)(metadata->getDevPtr()),
               sparseMatrixA->getDevPtr(), A_matrix->getRow(),
               A_matrix->getCol(), m_K, A_matrix->getNonzeroblocks(),
               A_matrix->getBlockInfo_dev(),
               A_matrix->getRelativeBlockIndexMapping_dev(),
               A_matrix->getMergedTileInfo_dev());

    return std::make_pair(std::move(metadata), std::move(sparseMatrixA));
  }

  template <typename Func, typename PreprocessFunc>
  void evaluateSparse24(Func &&hgemm, PreprocessFunc &&preprocess,
                        const std::string &name, bool large = false) {
    HLOG("----------------- Sparse Evaluating 24 %s -----------------",
         name.c_str());
    auto [metadata, sparseMatrixA] =
        getPreprocessed(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();
    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }
    // warm up
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(A_matrix->getBcsrValues(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), A_matrix->getRow(),
            m_C_for_sparse->getCol(), A_matrix->getCol(),
            A_matrix->getNonzeroblocks(), A_matrix->getBlockInfo_dev(),
            A_matrix->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    m_warmup_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                     (t2.tv_usec - t1.tv_usec) / 1000.0) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse24(std::forward<Func>(hgemm), preprocess, name, large);
  }

  template <typename Func2, typename PreprocessFunc>
  void evaluateSparse24_2(Func2 &&hgemm, PreprocessFunc &&preprocess,
                          const std::string &name, bool large = false) {
    HLOG("----------------- Sparse Evaluating 24 %s -----------------",
         name.c_str());
    auto [metadata, sparseMatrixA] =
        getPreprocessed(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();
    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }

    A_matrix->bcsrBlocking();
    // warm up
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(A_matrix->getBcsrValues(), A_matrix->getBcsrRowPtr(),
            A_matrix->getBcsrColIdx(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), A_matrix->getRow(),
            m_C_for_sparse->getCol(), A_matrix->getCol(),
            A_matrix->getNonzeroblocks(), A_matrix->getBlockInfo_dev(),
            A_matrix->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    m_warmup_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                     (t2.tv_usec - t1.tv_usec) / 1000.0) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse24_2(std::forward<Func2>(hgemm), preprocess, name, large);
  }
  template <typename Func2, typename PreprocessFunc>
  void evaluateSparse24_2_tiled(Func2 &&hgemm, PreprocessFunc &&preprocess,
                                const std::string &name, bool large = false) {
    HLOG("----------------- Sparse Evaluating 24 %s -----------------",
         name.c_str());
    auto [metadata, sparseMatrixA] =
        getPreprocessedTile(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();
    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }

    // A_matrix->bcsrBlocking();
    // warm up
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(A_matrix->getMergedBcsrValues(), A_matrix->getMergedBcsrRowPtr(),
            A_matrix->getMergedBcsrColIdx(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), A_matrix->getRow(),
            m_C_for_sparse->getCol(), A_matrix->getCol(),
            A_matrix->getNonzeroblocks(), A_matrix->getMergedBlockInfo_dev(),
            A_matrix->getMergedRelativeBlockIndexMapping_dev(),
            A_matrix->getMergedTileInfo_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    m_warmup_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                     (t2.tv_usec - t1.tv_usec) / 1000.0) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    if (m_enable_sparse_check) {
      m_C_for_sparse->moveToHost();
      m_C_for_sparse->checkValue(m_base_for_sparse);
    }

    profileSparse24_2_tiled(std::forward<Func2>(hgemm), preprocess, name,
                            large);
  }

  template <typename Func>
  void createLinePlot(Func &&hgemm, const std::string &name) {
    HLOG("----------------- Sparse Lineplot %s -----------------",
         name.c_str());
    // HLOG("%d", m_A_sparse->getBlockInfo_host()[0]);
    usleep(m_sleep_duration * 100);
    m_C_for_sparse->tearUp(m_base_for_sparse);

    // warm up
    m_cuda_timer.start();
    for (size_t i = 0; i < m_warmup_iterations; ++i) {
      hgemm(m_A_sparse->getBcsrValues(), m_A_sparse->getBcsrRowPtr(),
            m_A_sparse->getBcsrColIdx(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
    }
    m_warmup_time = static_cast<double>(m_cuda_timer.end()) /
                    static_cast<double>(m_warmup_iterations);
    HLOG("Warm up time: %.3f ms", m_warmup_time);

    for (int mult = 14; mult < 126; mult++) {
      m_B_for_sparse = new Matrix(m_K, mult * MMA_N, "Matrix B for sparse");

      m_C_for_sparse = new Matrix(m_M, mult * MMA_N, "Matrix C for sparse");
      m_C_for_sparse->memSetHost();
      m_C_for_sparse->moveToDevice();

      m_base = new Matrix(m_M, m_N, "Matrix Base");
      m_base_for_sparse = new Matrix(m_M, mult * MMA_N, "Matrix Base");

      CudaTimer *currentTimer = new CudaTimer();
      currentTimer->start();
      for (size_t i = 0; i < m_profiling_iterations; ++i) {
        hgemm(m_A_sparse->getBcsrValues(), m_A_sparse->getBcsrRowPtr(),
              m_A_sparse->getBcsrColIdx(), m_B_for_sparse->getDevPtr(),
              m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
              m_C_for_sparse->getCol(), m_A_sparse->getCol(),
              m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
              m_A_sparse->getRelativeBlockIndexMapping_dev());
      }
      m_profiling_time = static_cast<double>(currentTimer->end()) /
                         static_cast<double>(m_profiling_iterations);
      m_throughput = static_cast<double>(m_A_sparse->getNonzeroblocks() *
                                         MMA_M * MMA_K * 2) *
                     1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

      if ((std::abs(m_base_time) <= 1e-6) &&
          (std::abs(m_base_throughput) <= 1e-6)) {
        m_base_time = m_profiling_time;
        m_base_throughput = m_throughput;
      }
      FILE *fout;
      fout = fopen("results_smat.csv", "a");
      fprintf(fout, "%d, %lf\n", mult * MMA_N, m_profiling_time);
      fclose(fout);

      delete currentTimer;

      HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
           "(%.2f%%)",
           name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
           m_throughput, m_throughput / m_base_throughput * 100);
    }
  }

private:
  void cublas_tensor_op(half *A, half *B, half *C, size_t M, size_t N,
                        size_t K) {
    cublasHandle_t handle = nullptr;
    HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    half alpha = 1.0;
    half beta = 0.0;

    HGEMM_CHECK_CUBLAS_ERROR(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B,
                     CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template <typename Func> void profile(Func &&hgemm, const std::string &name) {
    m_cuda_timer.start();
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      hgemm(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_M, m_N,
            m_K);
    }
    m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
                       static_cast<double>(m_profiling_iterations);
    m_throughput = static_cast<double>(m_M * m_N * m_K * 2) * 1e-12 /
                   (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }

  template <typename Func>
  void profileSparse(Func &&hgemm, const std::string &name) {
    // m_cuda_timer.start();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      hgemm(m_A_sparse->getBcsrValues(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);

    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_throughput = static_cast<double>(m_A_sparse->getNonzeroblocks() * MMA_M *
                                       MMA_K * 2) *
                   1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s, %lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }

  template <typename Func>
  void profileSparse2(Func &&hgemm, const std::string &name) {
    struct timeval t1, t2;
    // m_cuda_timer.start();
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      // gettimeofday(&t1, NULL);
      hgemm(m_A_sparse->getBcsrValues(), m_A_sparse->getBcsrRowPtr(),
            m_A_sparse->getBcsrColIdx(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
      // cudaDeviceSynchronize();
      // gettimeofday(&t2, NULL);
      // m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec -
      // t1.tv_usec) / 1000.0); FILE* fout; fout = fopen("results_smat.csv",
      // "a"); fprintf(fout, "%lf\n", m_profiling_time); fclose(fout);
      HLOG("%lf", m_profiling_time);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);
    m_throughput = static_cast<double>(m_A_sparse->getNonzeroblocks() * MMA_M *
                                       MMA_K * 2) *
                   1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s,%lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }
  template <typename Func>
  void profileSparse2_tiled(Func &&hgemm, const std::string &name) {
    struct timeval t1, t2;
    // m_cuda_timer.start();
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      // gettimeofday(&t1, NULL);
      hgemm(
          m_A_sparse->getMergedBcsrValues(), m_A_sparse->getMergedBcsrRowPtr(),
          m_A_sparse->getMergedBcsrColIdx(), m_B_for_sparse->getDevPtr(),
          m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
          m_C_for_sparse->getCol(), m_A_sparse->getCol(),
          m_A_sparse->getNonzeroblocks(), m_A_sparse->getMergedBlockInfo_dev(),
          m_A_sparse->getMergedRelativeBlockIndexMapping_dev());
      // cudaDeviceSynchronize();
      // gettimeofday(&t2, NULL);
      // m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec -
      // t1.tv_usec) / 1000.0); FILE* fout; fout = fopen("results_smat.csv",
      // "a"); fprintf(fout, "%lf\n", m_profiling_time); fclose(fout);
      HLOG("%lf", m_profiling_time);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);
    m_throughput = static_cast<double>(m_A_sparse->getNonzeroblocks() * MMA_M *
                                       MMA_K * 2) *
                   1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s,%lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }

  template <typename Func, typename PreprocessFunc>
  void profileSparse24(Func &&hgemm, PreprocessFunc &&preprocess,
                       const std::string &name, bool large) {

    auto [metadata, sparseMatrixA] =
        getPreprocessed(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();

    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }

    // m_cuda_timer.start();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      hgemm(A_matrix->getBcsrValues(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), A_matrix->getRow(),
            m_C_for_sparse->getCol(), A_matrix->getCol(),
            A_matrix->getNonzeroblocks(), A_matrix->getBlockInfo_dev(),
            A_matrix->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);

    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    int k = 16;
    if (large) {
      k = 32;
    }
    m_throughput =
        static_cast<double>(A_matrix->getNonzeroblocks() * MMA_M * k * 2) *
        1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s, %lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }
  template <typename Func, typename PreprocessFunc>
  void profileSparse24_2(Func &&hgemm, PreprocessFunc &&preprocess,
                         const std::string &name, bool large) {

    auto [metadata, sparseMatrixA] =
        getPreprocessed(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();

    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }

    // m_cuda_timer.start();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      hgemm(m_A_sparse->getBcsrValues(), m_A_sparse->getBcsrRowPtr(),
            m_A_sparse->getBcsrColIdx(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), m_A_sparse->getRow(),
            m_C_for_sparse->getCol(), m_A_sparse->getCol(),
            m_A_sparse->getNonzeroblocks(), m_A_sparse->getBlockInfo_dev(),
            m_A_sparse->getRelativeBlockIndexMapping_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);

    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    int k = 16;
    if (large) {
      k = 32;
    }
    m_throughput =
        static_cast<double>(A_matrix->getNonzeroblocks() * MMA_M * k * 2) *
        1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s, %lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }
  template <typename Func, typename PreprocessFunc>
  void profileSparse24_2_tiled(Func &&hgemm, PreprocessFunc &&preprocess,
                               const std::string &name, bool large) {

    auto [metadata, sparseMatrixA] =
        getPreprocessedTile(std::forward<PreprocessFunc>(preprocess), large);
    cudaDeviceSynchronize();

    SparseMatrix *A_matrix = m_A_sparse;
    if (large) {
      A_matrix = m_A_sparse_large;
    }

    // m_cuda_timer.start();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (size_t i = 0; i < m_profiling_iterations; ++i) {
      hgemm(A_matrix->getMergedBcsrValues(), A_matrix->getMergedBcsrRowPtr(),
            A_matrix->getMergedBcsrColIdx(), (char *)(metadata->getDevPtr()),
            sparseMatrixA->getDevPtr(), m_B_for_sparse->getDevPtr(),
            m_C_for_sparse->getDevPtr(), A_matrix->getRow(),
            m_C_for_sparse->getCol(), A_matrix->getCol(),
            A_matrix->getNonzeroblocks(), A_matrix->getMergedBlockInfo_dev(),
            A_matrix->getMergedRelativeBlockIndexMapping_dev(),
            A_matrix->getMergedTileInfo_dev());
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    m_profiling_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0) /
                       static_cast<double>(m_profiling_iterations);

    // m_profiling_time = static_cast<double>(m_cuda_timer.end()) /
    // static_cast<double>(m_profiling_iterations);
    int k = 16;
    if (large) {
      k = 32;
    }
    m_throughput =
        static_cast<double>(A_matrix->getNonzeroblocks() * MMA_M * k * 2) *
        1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

    if ((std::abs(m_base_time) <= 1e-6) &&
        (std::abs(m_base_throughput) <= 1e-6)) {
      m_base_time = m_profiling_time;
      m_base_throughput = m_throughput;
    }

    FILE *fout;
    fout = fopen("results_smat.csv", "a");
    fprintf(fout, "%s, %lf\n", m_file.data(), m_profiling_time);
    fclose(fout);
    HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS "
         "(%.2f%%)",
         name.c_str(), m_profiling_time, m_profiling_time / m_base_time * 100,
         m_throughput, m_throughput / m_base_throughput * 100);
  }

  const size_t m_M = 512;
  const size_t m_N = 2048;
  const size_t m_K = 1024;
  const size_t m_warmup_iterations = 1;
  const size_t m_profiling_iterations = 10;
  const size_t m_sleep_duration = 10;
  const bool m_enable_check = false;
  const bool m_enable_sparse_check = false;
  const int m_N_mult = 1;
  std::string m_file;

  Matrix *m_A = nullptr; // row major, M * K
  Matrix *m_B = nullptr; // col major, K * N
  Matrix *m_C = nullptr; // row major, M * N
  Matrix *m_base =
      nullptr; // row major, M * N, base result, init matrix C before each hgemm

  SparseMatrix *m_A_sparse = nullptr;
  SparseMatrix *m_A_sparse_large = nullptr;
  Matrix *m_B_for_sparse = nullptr;
  Matrix *m_C_for_sparse = nullptr;
  Matrix *m_base_for_sparse = nullptr;

  CudaTimer m_cuda_timer;

  double m_warmup_time = 0.0;
  double m_profiling_time = 0.0;
  double m_throughput = 0.0;
  double m_base_time = 0.0;       // cublas tensor op default
  double m_base_throughput = 0.0; // cublas tensor op default

  HGEMM_DISALLOW_COPY_AND_ASSIGN(Tester);
};
