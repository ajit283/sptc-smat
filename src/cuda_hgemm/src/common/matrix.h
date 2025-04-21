#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"
#include "mmio_highlevel.h"
#include <malloc.h>

#define MMA_M 16
#define MMA_N 8
// #define MMA_K 32

#define BLOCK 4

class Matrix {
public:
  Matrix(size_t row, size_t col, std::string name = "Matrix", float min = -1.0,
         float max = 1.0)
      : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
    HGEMM_CHECK_GT(m_row, 0);
    HGEMM_CHECK_GT(m_col, 0);

    m_elem_num = m_row * m_col;
    HGEMM_CHECK_GT(m_elem_num, 0);

    m_host_ptr = new half[m_elem_num];
    HGEMM_CHECK(m_host_ptr);
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
    HGEMM_CHECK(m_dev_ptr);

    std::random_device rd;
    std::default_random_engine engine{rd()};
    std::uniform_real_distribution<float> uniform(m_min, m_max);
    for (size_t i = 0; i < m_elem_num; ++i) {
      // m_host_ptr[i] = __float2half(uniform(engine));
      m_host_ptr[i] = __float2half(1);
    }

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));

    HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col,
         m_host_ptr, m_dev_ptr);
  }

  ~Matrix() {
    if (m_host_ptr) {
      delete[] m_host_ptr;
      m_host_ptr = nullptr;
    }

    if (m_dev_ptr) {
      HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
      m_dev_ptr = nullptr;
    }
  }

  size_t getRow() const { return m_row; }

  size_t getCol() const { return m_col; }

  size_t getElemNum() const { return m_elem_num; }

  half *getHostPtr() const { return m_host_ptr; }

  half *getDevPtr() const { return m_dev_ptr; }

  void tearUp(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, base->getDevPtr(),
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToDevice));
  }

  void moveToHost() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToHost));
  }

  void moveToDevice() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));
  }

  void memSetHost() { memset(m_host_ptr, 0, m_elem_num * sizeof(half)); }

  void memSetDevice() {
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
  }

  void checkValue(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());

    m_max_diff = 0.0;
    m_avg_diff = 0.0;
    double diff = 0.0;
    int num_diff = 0;
    for (size_t i = 0; i < m_elem_num; ++i) {
      diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) -
                                          __half2float(base->getHostPtr()[i])));

      m_max_diff = std::max(m_max_diff, diff);
      m_avg_diff += diff;
      if (diff > 1e-5) {
        num_diff++;
      }

      // Print diff and values
      // printf("%.0f ", __half2float(m_host_ptr[i]));

      // // Add newline after every 64 values (assuming N=64)
      // if ((i + 1) % 64 == 0) {
      //   printf("\n");
      // }
    }
    m_avg_diff /= static_cast<double>(m_elem_num);

    HLOG("Max diff: %f, avg diff: %f, num diff: %i", m_max_diff, m_avg_diff,
         num_diff);
  }

private:
  const size_t m_row = 0;
  const size_t m_col = 0;
  const std::string m_name = "Matrix";
  // the threshold of the random matrix will affect the difference of the hgemm
  // results
  const float m_min = -1.0;
  const float m_max = 1.0;

  size_t m_elem_num = 0;
  half *m_host_ptr = nullptr;
  half *m_dev_ptr = nullptr;

  double m_max_diff = 0.0;
  double m_avg_diff = 0.0;

  HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};

class SparseMatrix {
public:
  SparseMatrix(const std::string &name = "Matrix",
               char *file = "./data/cop20k_A.mtx", int k = 16)
      : m_name(name), filename(file) {
    mMMA_K = k;

    readCsr();

    HLOG("Read %s", filename);
    // outputCsr();
    HGEMM_CHECK_GT(m_row, 0);
    HGEMM_CHECK_GT(m_col, 0);
    HGEMM_CHECK_GT(nnz, 0);

    HLOG("%zu x %zu, nnz = %zu, A[0] = %f", m_row, m_col, nnz,
         __half2float(csrVal_host[0]));

    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&csrVal_dev, nnz * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&csrColIdx_dev, nnz * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&csrRowPtr_dev, (m_row + 1) * sizeof(int)));

    HGEMM_CHECK(csrVal_dev);
    HGEMM_CHECK(csrColIdx_dev);
    HGEMM_CHECK(csrRowPtr_dev);

    // HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num *
    // sizeof(half), cudaMemcpyHostToDevice));

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        csrVal_dev, csrVal_host, nnz * sizeof(half), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(csrColIdx_dev, csrColIdx_host,
                                        nnz * sizeof(int),
                                        cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(csrRowPtr_dev, csrRowPtr_host,
                                        (m_row + 1) * sizeof(int),
                                        cudaMemcpyHostToDevice));

    csrToBcsr();
    bcsrBlocking();
    filterBcsrBlocks();
    // csrToBcsrKnapsacking();
    HLOG("Finished creating BCSR from CSR");
    HLOG("%zu total blocks, %zu nonzero blocks, %zu dense blocks, %zu sparse "
         "blocks",
         numberOfBlocks, nonzeroBlocks, denseBlocks, sparseBlocks);
    // HLOG("%d block is nonzero", blockInfo[0]);
    /* for (int i = 0; i < MMA_M; i++) {
        for (int j = 0; j < MMA_K; j++) {
            printf("%4.2f ", __half2float(bcsrVal_host[i * MMA_K + j]));
        }
        printf("\n");
    } */

    HGEMM_CHECK_CUDART_ERROR(cudaMalloc(
        (void **)&bcsrVal_dev, nonzeroBlocks * MMA_M * mMMA_K * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&bcsrColIdx_dev, nonzeroBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&bcsrRowPtr_dev,
                                        (m_row / MMA_M + 1) * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&blockInfo_dev, numberOfBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&relativeBlockIndexMapping_dev,
                                        numberOfBlocks * sizeof(int)));

    HGEMM_CHECK(bcsrVal_dev);
    HGEMM_CHECK(bcsrColIdx_dev);
    HGEMM_CHECK(bcsrRowPtr_dev);

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        bcsrVal_dev, bcsrVal_host,
        nonzeroBlocks * MMA_M * mMMA_K * sizeof(half), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(bcsrColIdx_dev, bcsrColIdx_host,
                                        nonzeroBlocks * sizeof(int),
                                        cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(bcsrRowPtr_dev, bcsrRowPtr_host,
                                        (m_row / MMA_M + 1) * sizeof(int),
                                        cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(blockInfo_dev, blockInfo_host,
                                        numberOfBlocks * sizeof(int),
                                        cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        relativeBlockIndexMapping_dev, relativeBlockIndexMapping_host,
        numberOfBlocks * sizeof(int), cudaMemcpyHostToDevice));
  }

  ~SparseMatrix() {
    // --- host pointers (free/delete in opposite order of alloc) ---
    if (m_host_ptr) {
      delete[] m_host_ptr;
      m_host_ptr = nullptr;
    }
    if (csrVal_host) {
      free(csrVal_host);
      csrVal_host = nullptr;
    }
    if (csrColIdx_host) {
      free(csrColIdx_host);
      csrColIdx_host = nullptr;
    }
    if (csrRowPtr_host) {
      free(csrRowPtr_host);
      csrRowPtr_host = nullptr;
    }

    if (bcsrVal_host) {
      free(bcsrVal_host);
      bcsrVal_host = nullptr;
    }
    if (bcsrColIdx_host) {
      free(bcsrColIdx_host);
      bcsrColIdx_host = nullptr;
    }
    if (bcsrRowPtr_host) {
      free(bcsrRowPtr_host);
      bcsrRowPtr_host = nullptr;
    }

    if (blockInfo_host) {
      free(blockInfo_host);
      blockInfo_host = nullptr;
    }
    if (relativeBlockIndexMapping_host) {
      free(relativeBlockIndexMapping_host);
      relativeBlockIndexMapping_host = nullptr;
    }

    if (sparseBcsrVal_host) {
      free(sparseBcsrVal_host);
      sparseBcsrVal_host = nullptr;
    }
    if (sparseBcsrColIdx_host) {
      free(sparseBcsrColIdx_host);
      sparseBcsrColIdx_host = nullptr;
    }
    if (sparseBcsrRowPtr_host) {
      free(sparseBcsrRowPtr_host);
      sparseBcsrRowPtr_host = nullptr;
    }

    if (denseBcsrVal_host) {
      free(denseBcsrVal_host);
      denseBcsrVal_host = nullptr;
    }
    if (denseBcsrColIdx_host) {
      free(denseBcsrColIdx_host);
      denseBcsrColIdx_host = nullptr;
    }
    if (denseBcsrRowPtr_host) {
      free(denseBcsrRowPtr_host);
      denseBcsrRowPtr_host = nullptr;
    }

    if (sparseBlockInfo_host) {
      free(sparseBlockInfo_host);
      sparseBlockInfo_host = nullptr;
    }
    if (sparseRelativeBlockIndexMapping_host) {
      free(sparseRelativeBlockIndexMapping_host);
      sparseRelativeBlockIndexMapping_host = nullptr;
    }

    if (mergedBcsrVal_host) {
      free(mergedBcsrVal_host);
      mergedBcsrVal_host = nullptr;
    }
    if (mergedBcsrColIdx_host) {
      free(mergedBcsrColIdx_host);
      mergedBcsrColIdx_host = nullptr;
    }
    if (mergedBcsrRowPtr_host) {
      free(mergedBcsrRowPtr_host);
      mergedBcsrRowPtr_host = nullptr;
    }

    if (mergedBlockInfo_host) {
      free(mergedBlockInfo_host);
      mergedBlockInfo_host = nullptr;
    }
    if (mergedRelativeBlockIndexMapping_host) {
      free(mergedRelativeBlockIndexMapping_host);
      mergedRelativeBlockIndexMapping_host = nullptr;
    }
    if (mergedTileInfo_host) {
      free(mergedTileInfo_host);
      mergedTileInfo_host = nullptr;
    }

    // --- device pointers (cudaFree everything you cudaMalloc’d) ---
    if (m_dev_ptr) {
      cudaFree(m_dev_ptr);
      m_dev_ptr = nullptr;
    }

    if (csrVal_dev) {
      cudaFree(csrVal_dev);
      csrVal_dev = nullptr;
    }
    if (csrColIdx_dev) {
      cudaFree(csrColIdx_dev);
      csrColIdx_dev = nullptr;
    }
    if (csrRowPtr_dev) {
      cudaFree(csrRowPtr_dev);
      csrRowPtr_dev = nullptr;
    }

    if (bcsrVal_dev) {
      cudaFree(bcsrVal_dev);
      bcsrVal_dev = nullptr;
    }
    if (bcsrColIdx_dev) {
      cudaFree(bcsrColIdx_dev);
      bcsrColIdx_dev = nullptr;
    }
    if (bcsrRowPtr_dev) {
      cudaFree(bcsrRowPtr_dev);
      bcsrRowPtr_dev = nullptr;
    }

    if (blockInfo_dev) {
      cudaFree(blockInfo_dev);
      blockInfo_dev = nullptr;
    }
    if (relativeBlockIndexMapping_dev) {
      cudaFree(relativeBlockIndexMapping_dev);
      relativeBlockIndexMapping_dev = nullptr;
    }

    if (sparseBcsrVal_dev) {
      cudaFree(sparseBcsrVal_dev);
      sparseBcsrVal_dev = nullptr;
    }
    if (sparseBcsrColIdx_dev) {
      cudaFree(sparseBcsrColIdx_dev);
      sparseBcsrColIdx_dev = nullptr;
    }
    if (sparseBcsrRowPtr_dev) {
      cudaFree(sparseBcsrRowPtr_dev);
      sparseBcsrRowPtr_dev = nullptr;
    }

    if (denseBcsrVal_dev) {
      cudaFree(denseBcsrVal_dev);
      denseBcsrVal_dev = nullptr;
    }
    if (denseBcsrColIdx_dev) {
      cudaFree(denseBcsrColIdx_dev);
      denseBcsrColIdx_dev = nullptr;
    }
    if (denseBcsrRowPtr_dev) {
      cudaFree(denseBcsrRowPtr_dev);
      denseBcsrRowPtr_dev = nullptr;
    }

    if (sparseBlockInfo_dev) {
      cudaFree(sparseBlockInfo_dev);
      sparseBlockInfo_dev = nullptr;
    }
    if (sparseRelativeBlockIndexMapping_dev) {
      cudaFree(sparseRelativeBlockIndexMapping_dev);
      sparseRelativeBlockIndexMapping_dev = nullptr;
    }

    if (mergedBcsrVal_dev) {
      cudaFree(mergedBcsrVal_dev);
      mergedBcsrVal_dev = nullptr;
    }
    if (mergedBcsrColIdx_dev) {
      cudaFree(mergedBcsrColIdx_dev);
      mergedBcsrColIdx_dev = nullptr;
    }
    if (mergedBcsrRowPtr_dev) {
      cudaFree(mergedBcsrRowPtr_dev);
      mergedBcsrRowPtr_dev = nullptr;
    }

    if (mergedBlockInfo_dev) {
      cudaFree(mergedBlockInfo_dev);
      mergedBlockInfo_dev = nullptr;
    }
    if (mergedRelativeBlockIndexMapping_dev) {
      cudaFree(mergedRelativeBlockIndexMapping_dev);
      mergedRelativeBlockIndexMapping_dev = nullptr;
    }
    if (mergedTileInfo_dev) {
      cudaFree(mergedTileInfo_dev);
      mergedTileInfo_dev = nullptr;
    }
  }
  size_t getRow() { return m_row; }

  size_t getCol() { return m_col; }

  size_t getElemNum() { return m_elem_num; }

  size_t getNnz() { return nnz; }

  half *getHostPtr() { return m_host_ptr; }

  half *getDevPtr() { return m_dev_ptr; }

  half *getBcsrValues() { return bcsrVal_dev; }

  int *getBcsrRowPtr() { return bcsrRowPtr_dev; }

  int *getBcsrColIdx() { return bcsrColIdx_dev; }

  size_t getSparseBlocks() { return sparseBlocks; }

  size_t getDenseBlocks() { return denseBlocks; }

  // Add to SparseMatrix class:
  half *getBcsrValuesHost() { return bcsrVal_host; }
  int *getBcsrRowPtrHost() { return bcsrRowPtr_host; }
  int *getBcsrColIdxHost() { return bcsrColIdx_host; }

  int *getRelativeBlockIndexMapping_dev() {
    return relativeBlockIndexMapping_dev;
  }

  size_t getNonzeroblocks() { return nonzeroBlocks; }

  int *getBlockInfo_dev() { return blockInfo_dev; }

  int *getBlockInfo_host() { return blockInfo_host; }

  half *getMergedBcsrValues() { return mergedBcsrVal_dev; }
  int *getMergedBcsrRowPtr() { return mergedBcsrRowPtr_dev; }
  int *getMergedBcsrColIdx() { return mergedBcsrColIdx_dev; }

  half *getMergedBcsrValuesHost() { return mergedBcsrVal_host; }
  int *getMergedBcsrRowPtrHost() { return mergedBcsrRowPtr_host; }
  int *getMergedBcsrColIdxHost() { return mergedBcsrColIdx_host; }

  // Add new getter methods
  int *getMergedRelativeBlockIndexMapping_dev() {
    return mergedRelativeBlockIndexMapping_dev;
  }

  int *getMergedRelativeBlockIndexMapping_host() {
    return mergedRelativeBlockIndexMapping_host;
  }

  int *getMergedBlockInfo_dev() { return mergedBlockInfo_dev; }

  int *getMergedBlockInfo_host() { return mergedBlockInfo_host; }

  int *getMergedTileInfo_host() { return mergedTileInfo_host; }

  int *getMergedTileInfo_dev() { return mergedTileInfo_dev; }

  size_t getMergedNonzeroBlocks() { return mergedNonzeroBlocks; }

  // Getters for 2:4 sparse BCSR blocks
  half *getSparseBcsrValues() { return sparseBcsrVal_dev; }
  int *getSparseBcsrRowPtr() { return sparseBcsrRowPtr_dev; }
  int *getSparseBcsrColIdx() { return sparseBcsrColIdx_dev; }

  // Getters for dense BCSR blocks
  half *getDenseBcsrValues() { return denseBcsrVal_dev; }
  int *getDenseBcsrRowPtr() { return denseBcsrRowPtr_dev; }
  int *getDenseBcsrColIdx() { return denseBcsrColIdx_dev; }

  // **** New getters for sparse block info and mapping ****
  int *getSparseBlockInfo_dev() { return sparseBlockInfo_dev; }
  int *getSparseRelativeBlockIndexMapping_dev() {
    return sparseRelativeBlockIndexMapping_dev;
  }

  // --- In the SparseMatrix class public section ---

  // Host getters for filtered sparse BCSR properties
  half *getSparseBcsrValuesHost() { return sparseBcsrVal_host; }
  int *getSparseBcsrRowPtrHost() { return sparseBcsrRowPtr_host; }
  int *getSparseBcsrColIdxHost() { return sparseBcsrColIdx_host; }

  // Host getter for the sparse block info array (all ones)
  int *getSparseBlockInfoHost() { return sparseBlockInfo_host; }

  // Host getter for the sparse relative block index mapping
  int *getSparseRelativeBlockIndexMappingHost() {
    return sparseRelativeBlockIndexMapping_host;
  }

  // Host getters for filtered dense BCSR properties
  half *getDenseBcsrValuesHost() { return denseBcsrVal_host; }
  int *getDenseBcsrRowPtrHost() { return denseBcsrRowPtr_host; }
  int *getDenseBcsrColIdxHost() { return denseBcsrColIdx_host; }

  void tearUp(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, base->getDevPtr(),
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToDevice));
  }

  void moveToHost() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToHost));
  }

  void moveToDevice() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));
  }

  void memSetHost() { memset(m_host_ptr, 0, m_elem_num * sizeof(half)); }

  void memSetDevice() {
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
  }

  void checkValue(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());
    if (m_elem_num == 0) {
      m_elem_num = m_row * m_col;
    }
    m_max_diff = 0.0;
    m_avg_diff = 0.0;
    double diff = 0.0;
    for (size_t i = 0; i < m_elem_num; ++i) {
      diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) -
                                          __half2float(base->getHostPtr()[i])));
      m_max_diff = std::max(m_max_diff, diff);
      m_avg_diff += diff;
    }

    m_avg_diff /= static_cast<double>(m_elem_num);

    HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
  }

  void readCsr() {
    int isSymmetric;
    mmio_allinone(&m_row, &m_col, &nnz, &isSymmetric, &csrRowPtr_host,
                  &csrColIdx_host, &csrVal_host, filename, mMMA_K);
  }

  void outputCsr() {
    // Open file_transformed.mtx for writing
    std::string file_name_str(filename);
    std::ofstream outfile("./data/magicube_cop20k_A.mtx");

    // Check if file is opened successfully
    if (!outfile) {
      std::cerr << "Failed to open file_transformed.mtx for writing."
                << std::endl;
      return; // Exit with error
    }

    // Write the values of a, b, c separated by commas in the first line
    outfile << m_row << ", " << m_col << ", " << nnz << std::endl;

    // Write the elements of array t separated by spaces in the second line
    for (int i = 0; i < m_row + 1; ++i) {
      outfile << csrRowPtr_host[i] << " ";
    }
    outfile << std::endl;

    for (int i = 0; i < nnz; ++i) {
      outfile << csrColIdx_host[i] << " ";
    }
    outfile << std::endl;
    // Close the file
    outfile.close();
    HLOG("Outputed to Magicube format.");
  }

  void makeDenseArray() {
    m_elem_num = m_row * m_col;
    m_host_ptr = new half[m_elem_num];
    HGEMM_CHECK(m_host_ptr);
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
    HGEMM_CHECK(m_dev_ptr);

    // fill everything with zeros
    for (size_t i = 0; i < m_elem_num; ++i) {
      m_host_ptr[i] = __float2half(0);
    }

    // fill with csr values
    for (size_t row = 0; row < m_row; row++) {
      for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) {
        size_t col = csrColIdx_host[j];
        half val = csrVal_host[j];
        m_host_ptr[row * m_col + col] = val;
      }
    }

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));
  }

  void csrToBcsr() {
    printf("mMMA_K %d\n", mMMA_K);
    // first prepare the info arrays
    size_t numColRegions = (m_col + mMMA_K - 1) / mMMA_K;
    size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;

    numberOfBlocks = numRowRegions * numColRegions;
    // printf("numblocks %d\n", numberOfBlocks);

    blockInfo_host = (int *)calloc(sizeof(int), numberOfBlocks);
    // 0 - zero block
    // 1 - 2:4 sparse block
    // 2 - dense block
    for (size_t row = 0; row < m_row; row++) {
      for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) {
        size_t col = csrColIdx_host[j];
        // printf("%f\n", csrValA[j]);
        // printf("col %d\n", col);
        size_t rowRegion = row / MMA_M;
        size_t colRegion = col / mMMA_K;
        // printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
        size_t blockIndex = rowRegion * numColRegions + colRegion;
        // printf("block  index %d\n", blockIndex);
        if (blockInfo_host[blockIndex] ==
            0) // zero block, stops being 0, becomes sparse
        {
          blockInfo_host[blockIndex] = 1;
          nonzeroBlocks += 1;
          sparseBlocks++;
        } else if (blockInfo_host[blockIndex] == 1) // sparse block
        {
          // check can it still be sparse
          // should I check previous two or I am in new part for 2:4
          size_t relative24Index = col % 4;
          if (relative24Index == 2 || relative24Index == 3) {
            if (j >= csrRowPtr_host[row] + 2 &&
                csrColIdx_host[j - 1] == col - 1 &&
                csrColIdx_host[j - 2] == col - 2) {
              blockInfo_host[blockIndex] = 2;
              denseBlocks++;
              sparseBlocks--;
            }
          }
        }
      }
    }

    size_t relativeIndex = 0;
    relativeBlockIndexMapping_host =
        (int *)malloc(numberOfBlocks * sizeof(int));
    for (size_t i = 0; i < numberOfBlocks; i++) {
      relativeBlockIndexMapping_host[i] =
          (blockInfo_host[i] != 0) ? relativeIndex++ : -1;
      // printf("relative [%d] = %d\n", i, relativeBlockIndexMapping[i]);
    }

    // get the bcsr
    bcsrRowPtr_host = (int *)calloc(sizeof(int), (m_row / MMA_M + 1));
    bcsrColIdx_host = (int *)malloc(nonzeroBlocks * sizeof(int));
    bcsrVal_host = (half *)calloc(sizeof(half), nonzeroBlocks * MMA_M * mMMA_K);

    size_t num_blocks = 0;

    // Do the rowPtrBcsr and colIdxBcsr
    for (size_t row = 0; row < m_row; row += MMA_M) {
      // printf("Problem in 314?\n");
      bcsrRowPtr_host[row / MMA_M] = num_blocks; // Update rowPtr
      // printf("rowPtr[%d] = %d\n", row/MMA_M, num_blocks);

      // Iterate through columns
      for (size_t col = 0; col < m_col; col += mMMA_K) {
        size_t current_block = row / MMA_M * numColRegions + col / mMMA_K;
        // printf("Problem in 320?");
        if (blockInfo_host[current_block] == 0) {
          continue;
        }
        // printf("Problem in 325?");
        bcsrColIdx_host[num_blocks] =
            col; // not relative bcsr columns index / MMA_K if want relative
        // printf("colIdx[%d] = %d\n", num_blocks, col);
        num_blocks++;
      }
    }

    // printf("Problem in 372?");
    bcsrRowPtr_host[m_row / MMA_M] = num_blocks; // Update last entry of rowPtr
    // printf("rowPtr[%d] = %d\n", numRows / MMA_M, num_blocks);

    // printf("%d total blocks\n", totalNumberOfBlocks);

    // Do the valuesBcsr
    for (size_t row = 0; row < m_row; row++) {
      for (size_t j = csrRowPtr_host[row]; j < csrRowPtr_host[row + 1]; j++) {
        size_t col = csrColIdx_host[j];
        // printf("col %d\n", col);
        size_t rowRegion = row / MMA_M;
        size_t colRegion = col / mMMA_K;
        // printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
        size_t blockIndex = rowRegion * numColRegions + colRegion;
        half val = csrVal_host[j];
        // printf("val %f\n", val);
        size_t offset = row % MMA_M * mMMA_K + col % mMMA_K;
        size_t bcsrIndex =
            relativeBlockIndexMapping_host[blockIndex] * MMA_M * mMMA_K +
            offset;
        // printf("relativeIndex %d x %d +  offset %d = %d\n",
        // relativeBlockIndexMapping[blockIndex], blockSize, offset, bcsrIndex);
        bcsrVal_host[bcsrIndex] = val;
      }
    }

    printf("bcsrRowPointer[0]: %d\n", bcsrRowPtr_host[0]);
    printf("bcsrRowPointer[1]: %d\n", bcsrRowPtr_host[1]);

    /* for (int i = 0; i < num_blocks * blockSize; i++)
    {
        if (i % blockSize == 0)
        {
            printf("\n");
        }
        printf("%f\n", *(*valuesBcsr + i));

    } */

    // create the data structures for locations of nnz blocks
    // not needed for now
  }

  void bcsrBlocking() {
    struct Block {
      int x;
      int y;
      half *vals;
      int *sparsity;
    };
    int size = nonzeroBlocks;
    std::vector<Block> blocks;
    blocks.reserve(size);

    // std::cout << "Starting with size: " << size << std::endl;

    size_t numColRegions = (m_col + (mMMA_K * BLOCK) - 1) / (mMMA_K * BLOCK);
    size_t numRowRegions = (m_row + (MMA_M * BLOCK) - 1) / (MMA_M * BLOCK);

    std::vector<std::vector<int>> mergedBlockInfo(
        numRowRegions, std::vector<int>(numColRegions, 0));

    for (int i = 0; i < size; i++) {
      // std::cout << "Processing i=" << i << std::endl;

      // int row = 0;
      // while (row < m_row / MMA_M && bcsrRowPtr_host[row + 1] <= i) {
      //   row++;
      // }

      int blockRow = 0;
      while (blockRow < m_row / MMA_M && bcsrRowPtr_host[blockRow + 1] <= i) {
        blockRow++;
      }
      int row = blockRow * MMA_M; // Convert block row to actual row number

      // std::cout << "Found row: " << row << std::endl;
      // std::cout << "Accessing colIdx at " << i << std::endl;

      int col = bcsrColIdx_host[i];
      // std::cout << "Col is: " << col << std::endl;

      int aligned_x = col - (col % (BLOCK * mMMA_K));
      int aligned_y = row - (row % (BLOCK * MMA_M));
      bool partOfPreviousBlock = false;

      // check if there is a previous block that this would be part of
      for (auto e : blocks) {
        if (e.x == aligned_x && e.y == aligned_y) {
          {
            partOfPreviousBlock = true;
            // std::cout << "col: " << col << " e.x: " << e.x
            //           << " diff: " << (col - e.x)
            //           << " mod: " << ((col - e.x) % MMA_K) << std::endl;
            // int positionInBlock_x = (col - e.x) - ((col - e.x) % MMA_K);
            // int positionInBlock_y = (row - e.y) - ((row - e.y) % MMA_M);
            int positionInBlock_x = (col - e.x);
            int positionInBlock_y = (row - e.y);
            // std::cout << "Position x: " << positionInBlock_x <<
            // std::endl; std::cout << "Position y: " << positionInBlock_y
            // << std::endl; std::cout << "Dest offset: "
            //           << MMA_K * BLOCK * positionInBlock_y +
            //                  positionInBlock_x * MMA_M
            //           << std::endl;
            // std::cout << "Source offset: " << i * MMA_M * MMA_K <<
            // std::endl;

            memcpy(e.vals + mMMA_K * BLOCK * positionInBlock_y +
                       positionInBlock_x * MMA_M,
                   bcsrVal_host + i * MMA_M * mMMA_K,
                   sizeof(half) * MMA_M * mMMA_K);

            e.sparsity[((positionInBlock_y) / MMA_M) * BLOCK +
                       (positionInBlock_x) / mMMA_K] = 1;

            // Print some values we just copied
            // for (int j = 0; j < 5; j++) {
            //   std::cout << "Value at " << j << ": "
            //             << __half2float(
            //                    e.vals[MMA_K * BLOCK * positionInBlock_y +
            //                           positionInBlock_x + j])
            //             << "\n";
            // }
            // std::cout << "done" << std::endl;
            break;
          }
        }
      }
      if (!partOfPreviousBlock) {
        // std::cout << "Creating new block at (" << aligned_y << "," <<
        // aligned_x
        //           << ")\n";
        int positionInBlock_x = (col % (BLOCK * mMMA_K));
        int positionInBlock_y = (row % (BLOCK * MMA_M));
        // std::cout << "Position in block: (" << positionInBlock_y << ","
        //           << positionInBlock_x << ")\n";

        half *vals =
            (half *)calloc(BLOCK * BLOCK * MMA_M * mMMA_K, sizeof(half));
        size_t dest_offset =
            positionInBlock_y * mMMA_K * BLOCK + positionInBlock_x * MMA_M;
        size_t src_offset = i * MMA_M * mMMA_K;
        // std::cout << "Copying " << MMA_M * MMA_K << " values from offset "
        //           << src_offset << " to offset " << dest_offset << "\n";
        memcpy(vals + dest_offset, bcsrVal_host + src_offset,
               MMA_M * mMMA_K * sizeof(half));

        mergedBlockInfo.at(aligned_y / (BLOCK * MMA_M))
            .at(aligned_x / (BLOCK * mMMA_K)) = 1;

        // Print some values we just copied
        // for (int j = 0; j < 5; j++) {
        //   std::cout << "Value at " << j << ": "
        //             << __half2float(vals[dest_offset + j]) << "\n";
        // }

        int *sparsity = (int *)calloc(BLOCK * BLOCK, sizeof(int));

        sparsity[(positionInBlock_y / MMA_M) * BLOCK +
                 (positionInBlock_x / mMMA_K)] = 1;

        Block block = {aligned_x, aligned_y, vals, sparsity};
        blocks.push_back(block);
      }
      // for (int b = 0; b < blocks.size(); b++) {
      //   // std::cout << "Block " << b << " after iteration:\n";
      //   for (int j = 0; j < BLOCK * BLOCK * MMA_M * MMA_K; j++) {
      //     std::cout << __half2float(blocks[b].vals[j]) << " ";
      //   }
      //   std::cout << "\n";
      // }
      // std::cout << "Block pointer address for block " << blocks.size() - 1
      //           << ": " << (void *)blocks.back().vals << "\n";

      // std::cout << "-----------------------------" << std::endl;
    }

    // print each block
    // for (int i = 0; i < blocks.size(); i++) {
    //   std::cout << "Block " << i << ":\n";
    //   for (int j = 0; j < BLOCK * BLOCK * MMA_M * MMA_K; j++) {
    //     std::cout << __half2float(blocks[i].vals[j]) << " ";
    //   }
    //   std::cout << "\n";
    // }

    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<half> values;
    std::vector<half> sparsity_tile;

    std::vector<std::vector<int>> mergedRelativeBlockIndexMapping(
        numRowRegions, std::vector<int>(numColRegions, 0));

    int relativeIndex = 0;

    for (int y = 0; y < numRowRegions; y++) {
      for (int x = 0; x < numColRegions; x++) {
        mergedRelativeBlockIndexMapping.at(y).at(x) =
            ((mergedBlockInfo.at(y).at(x) != 0) ? relativeIndex++ : -1);
      }
    }

    // Initialize rowPtr
    // rowPtr.push_back(0);
    int currentRow = -1;

    // Build BCSR format
    for (int i = 0; i < blocks.size(); i++) {
      // New row detection
      if (currentRow != blocks[i].y) {
        currentRow = blocks[i].y;
        rowPtr.push_back(i); // Start index of this row
      }

      // Add column index
      colIdx.push_back(blocks[i].x);

      // Add values
      for (int j = 0; j < BLOCK * BLOCK * MMA_M * mMMA_K; j++) {
        values.push_back(blocks[i].vals[j]);
      }
    }

    for (int i = 0; i < blocks.size(); i++) {
      for (int j = 0; j < BLOCK * BLOCK; j++) {
        sparsity_tile.push_back(blocks[i].sparsity[j]);
      }
    }
    // Add final rowPtr entry
    rowPtr.push_back(blocks.size());
    // rowPtr.push_back(blocks.size());
    // rowPtr.push_back(blocks.size());
    // rowPtr.push_back(blocks.size());
    // rowPtr.push_back(blocks.size());
    // rowPtr.push_back(blocks.size());

    mergedNonzeroBlocks = blocks.size();

    // Allocate host memory
    mergedBcsrVal_host = (half *)calloc(values.size(), sizeof(half));
    mergedBcsrRowPtr_host = (int *)malloc(rowPtr.size() * sizeof(int));
    mergedBcsrColIdx_host = (int *)malloc(colIdx.size() * sizeof(int));

    // Copy data to host arrays
    memcpy(mergedBcsrVal_host, values.data(), values.size() * sizeof(half));
    memcpy(mergedBcsrRowPtr_host, rowPtr.data(), rowPtr.size() * sizeof(int));
    memcpy(mergedBcsrColIdx_host, colIdx.data(), colIdx.size() * sizeof(int));

    // Allocate device memory
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&mergedBcsrVal_dev, values.size() * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&mergedBcsrRowPtr_dev,
                                        rowPtr.size() * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&mergedBcsrColIdx_dev,
                                        colIdx.size() * sizeof(int)));

    // Copy to device
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(mergedBcsrVal_dev, mergedBcsrVal_host,
                                        values.size() * sizeof(half),
                                        cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(mergedBcsrRowPtr_dev, mergedBcsrRowPtr_host,
                   rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(mergedBcsrColIdx_dev, mergedBcsrColIdx_host,
                   colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));

    mergedBlockInfo_host =
        (int *)malloc(numRowRegions * numColRegions * sizeof(int));
    mergedRelativeBlockIndexMapping_host =
        (int *)malloc(numRowRegions * numColRegions * sizeof(int));

    // --- Replace the old mergedTileInfo_host assignment with the following
    // code ---
    // Previously:
    // mergedTileInfo_host = (int *)malloc(blocks.size() * BLOCK * BLOCK *
    // sizeof(int)); memcpy(mergedTileInfo_host, sparsity_tile.data(),
    // sparsity_tile.size() * sizeof(int));

    // New code: compute mergedTileInfo per subblock (each subblock is MMA_M x
    // mMMA_K, e.g., 16x32)
    mergedTileInfo_host =
        (int *)malloc(blocks.size() * BLOCK * BLOCK * sizeof(int));
    int mergedBlocks = blocks.size();

    for (int i = 0; i < mergedBlocks; i++) {
      // Each merged block has dimensions (BLOCK * MMA_M) x (BLOCK * mMMA_K)
      // Get pointer to the start of this merged block in mergedBcsrVal_host.
      half *mergedBlockVal =
          mergedBcsrVal_host + i * (BLOCK * MMA_M) * (BLOCK * mMMA_K);

      // Iterate over each subblock (tile) within the merged block.
      // Subblocks are arranged in a BLOCK x BLOCK grid.
      for (int sb_r = 0; sb_r < BLOCK; sb_r++) {
        for (int sb_c = 0; sb_c < BLOCK; sb_c++) {
          bool allZero = true;
          bool followsTwoFour = true;

          // Compute the starting row and column (within the merged block) for
          // this subblock.
          int subblockStartRow = sb_r * MMA_M;
          int subblockStartCol = sb_c * mMMA_K;

          // Process each row in the subblock.
          for (int r = 0; r < MMA_M; r++) {
            // The row in the merged block:
            int rowIdx = subblockStartRow + r;
            // Pointer to the beginning of this row in the merged block.
            half *rowPtr = mergedBlockVal + rowIdx * (BLOCK * mMMA_K);

            // Process the row in groups of 4 (assumes mMMA_K is divisible by 4)
            for (int g = 0; g < mMMA_K / 4; g++) {
              int nonzeroCount = 0;
              for (int c = 0; c < 4; c++) {
                half val = rowPtr[subblockStartCol + g * 4 + c];
                // A simple nonzero check; adjust if your half type requires a
                // different test.
                if (val != __float2half(0.0f)) {
                  nonzeroCount++;
                }
              }
              // If any group has nonzero entries, the row is not all zero.
              if (nonzeroCount > 0) {
                allZero = false;
              }
              // In 2:4 sparsity a nonzero group must have exactly 2 nonzeros.
              if (nonzeroCount != 0 && nonzeroCount > 2) {
                followsTwoFour = false;
              }
            }
          }

          // Determine the subblock type.
          // 0: entirely zero, 1: 2:4 sparse, 2: dense (not following 2:4)
          int tileType = 0;
          if (!allZero) {
            tileType = followsTwoFour ? 1 : 2;
          }

          // Save in row-major order: the subblock index is (sb_r * BLOCK +
          // sb_c)
          mergedTileInfo_host[i * (BLOCK * BLOCK) + (sb_r * BLOCK + sb_c)] =
              tileType;
        }
      }
    }

    // Copy from vectors to flat arrays
    for (size_t i = 0; i < numRowRegions; i++) {
      for (size_t j = 0; j < numColRegions; j++) {
        mergedBlockInfo_host[i * numColRegions + j] = mergedBlockInfo[i][j];
        mergedRelativeBlockIndexMapping_host[i * numColRegions + j] =
            mergedRelativeBlockIndexMapping[i][j];
      }
    }

    // Allocate device memory
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&mergedBlockInfo_dev,
                   numRowRegions * numColRegions * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&mergedRelativeBlockIndexMapping_dev,
                   numRowRegions * numColRegions * sizeof(int)));

    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&mergedTileInfo_dev,
                   blocks.size() * BLOCK * BLOCK * sizeof(int)));

    // Copy to device
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        mergedBlockInfo_dev, mergedBlockInfo_host,
        numRowRegions * numColRegions * sizeof(int), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        mergedRelativeBlockIndexMapping_dev,
        mergedRelativeBlockIndexMapping_host,
        numRowRegions * numColRegions * sizeof(int), cudaMemcpyHostToDevice));

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(
        mergedTileInfo_dev, mergedTileInfo_host,
        blocks.size() * BLOCK * BLOCK * sizeof(int), cudaMemcpyHostToDevice));

    // Free block memory
    for (auto &block : blocks) {
      free(block.vals);
    }
  }
  // -----

  void filterBcsrBlocks() {
    // Determine the number of block regions.
    int numRowRegions =
        m_row / MMA_M; // Adjust if m_row isn't an exact multiple.
    int numColRegions = (m_col + mMMA_K - 1) / mMMA_K;

    // Create temporary row pointer vectors for sparse and dense blocks.
    std::vector<int> sparseRowPtr(numRowRegions + 1, 0);
    std::vector<int> denseRowPtr(numRowRegions + 1, 0);

    // First pass: count blocks per block row.
    for (int r = 0; r < numRowRegions; r++) {
      int sparseCount = 0;
      int denseCount = 0;
      for (int c = 0; c < numColRegions; c++) {
        int global_index = r * numColRegions + c;
        if (blockInfo_host[global_index] == 1) {
          sparseCount++;
        } else if (blockInfo_host[global_index] == 2) {
          denseCount++;
        }
      }
      sparseRowPtr[r + 1] = sparseRowPtr[r] + sparseCount;
      denseRowPtr[r + 1] = denseRowPtr[r] + denseCount;
    }

    int totalSparseBlocks = sparseRowPtr[numRowRegions];
    int totalDenseBlocks = denseRowPtr[numRowRegions];

    // Allocate host arrays for the filtered sparse blocks.
    sparseBcsrRowPtr_host = (int *)malloc((numRowRegions + 1) * sizeof(int));
    sparseBcsrColIdx_host = (int *)malloc(totalSparseBlocks * sizeof(int));
    sparseBcsrVal_host =
        (half *)calloc(totalSparseBlocks * MMA_M * mMMA_K, sizeof(half));

    // Allocate host arrays for the filtered dense blocks.
    denseBcsrRowPtr_host = (int *)malloc((numRowRegions + 1) * sizeof(int));
    denseBcsrColIdx_host = (int *)malloc(totalDenseBlocks * sizeof(int));
    denseBcsrVal_host =
        (half *)calloc(totalDenseBlocks * MMA_M * mMMA_K, sizeof(half));

    // Copy the row pointer vectors to our host arrays.
    memcpy(sparseBcsrRowPtr_host, sparseRowPtr.data(),
           (numRowRegions + 1) * sizeof(int));
    memcpy(denseBcsrRowPtr_host, denseRowPtr.data(),
           (numRowRegions + 1) * sizeof(int));

    // Create counters for each block row.
    std::vector<int> sparseRowCounter(numRowRegions, 0);
    std::vector<int> denseRowCounter(numRowRegions, 0);

    // Second pass: fill in column indices and block values.
    for (int r = 0; r < numRowRegions; r++) {
      for (int c = 0; c < numColRegions; c++) {
        int global_index = r * numColRegions + c;
        int relIndex = relativeBlockIndexMapping_host[global_index];
        if (relIndex < 0)
          continue; // Skip zero blocks.

        if (blockInfo_host[global_index] == 1) { // 2:4 sparse block.
          int idx = sparseRowPtr[r] + sparseRowCounter[r];
          sparseBcsrColIdx_host[idx] = bcsrColIdx_host[relIndex];
          memcpy(sparseBcsrVal_host + idx * MMA_M * mMMA_K,
                 bcsrVal_host + relIndex * MMA_M * mMMA_K,
                 MMA_M * mMMA_K * sizeof(half));
          sparseRowCounter[r]++;
        } else if (blockInfo_host[global_index] == 2) { // Dense block.
          int idx = denseRowPtr[r] + denseRowCounter[r];
          denseBcsrColIdx_host[idx] = bcsrColIdx_host[relIndex];
          memcpy(denseBcsrVal_host + idx * MMA_M * mMMA_K,
                 bcsrVal_host + relIndex * MMA_M * mMMA_K,
                 MMA_M * mMMA_K * sizeof(half));
          denseRowCounter[r]++;
        }
      }
    }

    // Allocate and copy device memory for the sparse blocks.
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&sparseBcsrVal_dev,
                   totalSparseBlocks * MMA_M * mMMA_K * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&sparseBcsrRowPtr_dev,
                                        (numRowRegions + 1) * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&sparseBcsrColIdx_dev,
                                        totalSparseBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(sparseBcsrVal_dev, sparseBcsrVal_host,
                   totalSparseBlocks * MMA_M * mMMA_K * sizeof(half),
                   cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(sparseBcsrRowPtr_dev, sparseBcsrRowPtr_host,
                   (numRowRegions + 1) * sizeof(int), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(sparseBcsrColIdx_dev, sparseBcsrColIdx_host,
                   totalSparseBlocks * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and copy device memory for the dense blocks.
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&denseBcsrVal_dev,
                   totalDenseBlocks * MMA_M * mMMA_K * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&denseBcsrRowPtr_dev,
                                        (numRowRegions + 1) * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&denseBcsrColIdx_dev,
                                        totalDenseBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(denseBcsrVal_dev, denseBcsrVal_host,
                   totalDenseBlocks * MMA_M * mMMA_K * sizeof(half),
                   cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(denseBcsrRowPtr_dev, denseBcsrRowPtr_host,
                   (numRowRegions + 1) * sizeof(int), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(denseBcsrColIdx_dev, denseBcsrColIdx_host,
                   totalDenseBlocks * sizeof(int), cudaMemcpyHostToDevice));

    // **** Create a sparse block info array (for filtered sparse blocks) ****
    // Since all blocks here are 2:4 sparse, we simply fill with 1's.
    sparseBlockInfo_host = (int *)malloc(totalSparseBlocks * sizeof(int));
    for (int i = 0; i < totalSparseBlocks; i++) {
      sparseBlockInfo_host[i] = 1;
    }
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&sparseBlockInfo_dev,
                                        totalSparseBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemcpy(sparseBlockInfo_dev, sparseBlockInfo_host,
                   totalSparseBlocks * sizeof(int), cudaMemcpyHostToDevice));

    // **** Create a relative block index mapping for only the sparse blocks
    // **** This mapping is defined over the global block grid.
    int totalGlobalBlocks = numRowRegions * numColRegions;
    sparseRelativeBlockIndexMapping_host =
        (int *)malloc(totalGlobalBlocks * sizeof(int));
    int sparseCounter = 0;
    for (int i = 0; i < totalGlobalBlocks; i++) {
      if (blockInfo_host[i] == 1) {
        sparseRelativeBlockIndexMapping_host[i] = sparseCounter;
        sparseCounter++;
      } else {
        sparseRelativeBlockIndexMapping_host[i] = -1;
      }
    }
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&sparseRelativeBlockIndexMapping_dev,
                   totalGlobalBlocks * sizeof(int)));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(sparseRelativeBlockIndexMapping_dev,
                                        sparseRelativeBlockIndexMapping_host,
                                        totalGlobalBlocks * sizeof(int),
                                        cudaMemcpyHostToDevice));
  }

  void csrToBcsrKnapsacking() {
    size_t numColRegions = (m_col + mMMA_K - 1) / mMMA_K;
    size_t numRowRegions = (m_row + MMA_M - 1) / MMA_M;
    std::vector<std::vector<std::tuple<size_t, size_t, float, long, size_t>>>
        startsEndsOfCols(numRowRegions);

    bcsrRowPtr_host = (int *)calloc(sizeof(int), (m_row / MMA_M + 1));
    std::vector<size_t> vecOfColIdx;
    for (size_t row = 0; row < m_row; row += MMA_M) {
      bcsrRowPtr_host[row / MMA_M] = nonzeroBlocks;
      // printf("[%lu] = %d\n", (unsigned long) row / MMA_M, nonzeroBlocks);
      std::vector<size_t> columnsInBlock;
      for (size_t pointer = row; pointer < row + MMA_M; pointer++) {
        // dodaj iteraciju po columnima
        for (size_t j = csrRowPtr_host[pointer];
             j < csrRowPtr_host[pointer + 1]; j++) {
          // columnsInBlock.push_back(csrColIdx_host[j]);
          startsEndsOfCols[row / MMA_M].push_back(std::make_tuple(
              csrColIdx_host[j], pointer, __half2float(csrVal_host[j]), -1, 0));
        }
      }
      // std::sort(columnsInBlock.begin(), columnsInBlock.end());
      std::sort(
          startsEndsOfCols[row / MMA_M].begin(),
          startsEndsOfCols[row / MMA_M].end(),
          [](const std::tuple<size_t, size_t, float, long, size_t> &tuple1,
             const std::tuple<size_t, size_t, float, long, size_t> &tuple2) {
            return std::get<0>(tuple1) < std::get<0>(tuple2);
          });

      for (int i = 0; i < startsEndsOfCols[row / MMA_M].size(); i++) {
        auto a = startsEndsOfCols[row / MMA_M][i];
        // printf("%lu %lu %f %ld %lu\n", std::get<0>(a), std::get<1>(a),
        // std::get<2>(a), std::get<3>(a), std::get<4>(a));
      }

      // if (columnsInBlock.empty()) {
      if (startsEndsOfCols[row / MMA_M].empty()) {
        continue;
      }

      /* size_t start = columnsInBlock[0];
      size_t end = start + MMA_K;
      startsEndsOfCols[row / MMA_M].push_back(start);
      startsEndsOfCols[row / MMA_M].push_back(end);
      nonzeroBlocks++;
      for (size_t i = 1; i < columnsInBlock.size(); ++i) {
          if (columnsInBlock[i] >= end) {
              start = columnsInBlock[i];
              end = start + MMA_K;
              startsEndsOfCols[row / MMA_M].push_back(start);
              startsEndsOfCols[row / MMA_M].push_back(end);
              nonzeroBlocks++;
          }
      } */

      size_t firstColumn = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
      size_t lastColumn = std::get<0>(startsEndsOfCols[row / MMA_M].back());
      size_t start;
      size_t span = lastColumn - firstColumn + 1;
      size_t potentialAddLeft = firstColumn - 0;
      size_t potentialAddRight = m_col - lastColumn - 1;
      size_t to_add = mMMA_K - span % mMMA_K;

      if (span % mMMA_K == 0) {
        start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
      } else {

        if (potentialAddRight >= to_add) {
          start = std::get<0>(startsEndsOfCols[row / MMA_M][0]);
        } else if (potentialAddLeft >= to_add) {
          start = std::get<0>(startsEndsOfCols[row / MMA_M][0]) - to_add;
        } else {
          start =
              std::get<0>(startsEndsOfCols[row / MMA_M][0]) - potentialAddLeft;
        }
      }

      vecOfColIdx.push_back(start);
      size_t end = start + mMMA_K;
      std::get<3>(startsEndsOfCols[row / MMA_M][0]) = nonzeroBlocks;
      // change relative column
      std::get<4>(startsEndsOfCols[row / MMA_M][0]) =
          std::get<0>(startsEndsOfCols[row / MMA_M][0]) - start;
      // nonzeroBlocks++;
      /* auto a = startsEndsOfCols[row / MMA_M][0];
      printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a),
      std::get<2>(a), std::get<3>(a)); printf("span %lu addLeft %lu addRight
      %lu toAdd %lu ", span, potentialAddLeft, potentialAddRight, to_add);
      printf(" start  %lu end %lu \n", start, end); */
      for (size_t i = 1; i < startsEndsOfCols[row / MMA_M].size(); ++i) {
        // a = startsEndsOfCols[row / MMA_M][i];
        // printf("%lu %lu %f %ld    ==>\n", std::get<0>(a), std::get<1>(a),
        // std::get<2>(a), std::get<3>(a));
        if (std::get<0>(startsEndsOfCols[row / MMA_M][i]) >= end) {
          // printf("tusam\n");
          span = lastColumn - std::get<0>(startsEndsOfCols[row / MMA_M][i]) + 1;
          potentialAddLeft =
              std::get<0>(startsEndsOfCols[row / MMA_M][i]) - end;
          to_add = mMMA_K - span % mMMA_K;
          // printf("span %lu addLeft %lu addRight %lu toAdd %lu ", span,
          // potentialAddLeft, potentialAddRight, to_add);
          if (span % mMMA_K == 0) {
            start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
          } else {
            if (potentialAddRight >= to_add) {
              start = std::get<0>(startsEndsOfCols[row / MMA_M][i]);
            } else if (potentialAddLeft >= to_add) {
              start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) - to_add;
            } else {
              start = std::get<0>(startsEndsOfCols[row / MMA_M][i]) -
                      potentialAddLeft;
            }
          }

          vecOfColIdx.push_back(start);
          end = start + mMMA_K;
          // printf(" start  %lu end %lu \n", start, end);
          nonzeroBlocks++;
        }

        // printf("cols %lu %lu %lu\n", (unsigned long) start, (unsigned
        // long) end, (unsigned long) nonzeroBlocks);
        std::get<3>(startsEndsOfCols[row / MMA_M][i]) = nonzeroBlocks;
        std::get<4>(startsEndsOfCols[row / MMA_M][i]) =
            std::get<0>(startsEndsOfCols[row / MMA_M][i]) - start;
      }
      nonzeroBlocks++;
    }

    bcsrRowPtr_host[m_row / MMA_M] = nonzeroBlocks;

    // printf ("%lu nnzblocks\n", (unsigned long) nonzeroBlocks);
    bcsrColIdx_host = (int *)malloc(nonzeroBlocks * sizeof(int));
    bcsrVal_host = (half *)calloc(sizeof(half), nonzeroBlocks * MMA_M * mMMA_K);

    size_t current_idx = 0;
    for (size_t i = 0; i < nonzeroBlocks; i++) {
      bcsrColIdx_host[i] = vecOfColIdx[i];
      // printf("%d ", bcsrColIdx_host[i]);
    }

    // printf("\n");
    for (size_t rowRegion = 0; rowRegion < startsEndsOfCols.size();
         rowRegion++) {
      for (size_t element = 0; element < startsEndsOfCols[rowRegion].size();
           element++) {
        size_t col = std::get<0>(startsEndsOfCols[rowRegion][element]);
        size_t row = std::get<1>(startsEndsOfCols[rowRegion][element]);
        half val =
            __float2half(std::get<2>(startsEndsOfCols[rowRegion][element]));
        size_t relBlock = std::get<3>(startsEndsOfCols[rowRegion][element]);
        size_t relColumn = std::get<4>(startsEndsOfCols[rowRegion][element]);
        bcsrVal_host[relBlock * MMA_M * mMMA_K + (row % MMA_M) * mMMA_K +
                     relColumn] = val;
        // printf("%lu %lu %f %ld %lu ==> %lu\n", row, col,
        // __half2float(val), relBlock, relColumn, relBlock * MMA_M * MMA_K
        // + (row % MMA_M) * MMA_K
        // + relColumn);
      }
      // printf("\n");
    }

    /* printf("\nrowptr\n");
    //printf("%d rowptr size", (int)m_row / MMA_M + 1);
    for (int i = 0; i < m_row / MMA_M + 1; i++) {
        printf("%d ", bcsrRowPtr_host[i]);
    }
    printf("\n");

    for (int i = 0; i < nonzeroBlocks * MMA_M * MMA_K; i++) {

        if (i % (MMA_K * MMA_M) == 0) {
            printf ("\n\n");
        }
        else if (i % MMA_K == 0) {
            printf("\n");

        }
        printf("%f ", __half2float(bcsrVal_host[i]));
    } */
  }

private:
  size_t m_row = 0;
  size_t m_col = 0;
  std::string m_name = "Matrix";
  // the threshold of the random matrix will affect the difference of the
  // hgemm results
  float m_min = -1.0;
  float m_max = 1.0;

  size_t m_elem_num = 0;
  half *m_host_ptr = nullptr;
  half *m_dev_ptr = nullptr;

  double m_max_diff = 0.0;
  double m_avg_diff = 0.0;

  HGEMM_DISALLOW_COPY_AND_ASSIGN(SparseMatrix);

  char *filename = "";

  int mMMA_K = 16;

  size_t nnz = 0;

  half *csrVal_host = nullptr;
  int *csrRowPtr_host = nullptr;
  int *csrColIdx_host = nullptr;

  half *bcsrVal_host = nullptr;
  int *bcsrRowPtr_host = nullptr;
  int *bcsrColIdx_host = nullptr;

  // New members for merged blocks
  half *mergedBcsrVal_host = nullptr;
  int *mergedBcsrRowPtr_host = nullptr;
  int *mergedBcsrColIdx_host = nullptr;

  half *mergedBcsrVal_dev = nullptr;
  int *mergedBcsrRowPtr_dev = nullptr;
  int *mergedBcsrColIdx_dev = nullptr;

  size_t mergedNonzeroBlocks = 0;

  int *mergedBlockInfo_host = nullptr;
  int *mergedRelativeBlockIndexMapping_host = nullptr;

  int *mergedTileInfo_host = nullptr;

  int *mergedBlockInfo_dev = nullptr;
  int *mergedRelativeBlockIndexMapping_dev = nullptr;

  int *mergedTileInfo_dev = nullptr;

  half *csrVal_dev = nullptr;
  int *csrRowPtr_dev = nullptr;
  int *csrColIdx_dev = nullptr;

  half *bcsrVal_dev = nullptr;
  int *bcsrRowPtr_dev = nullptr;
  int *bcsrColIdx_dev = nullptr;

  int *blockInfo_host = nullptr;
  int *relativeBlockIndexMapping_host = nullptr;

  int *blockInfo_dev = nullptr;
  int *relativeBlockIndexMapping_dev = nullptr;

  half *sparseBcsrVal_host = nullptr;
  int *sparseBcsrRowPtr_host = nullptr;
  int *sparseBcsrColIdx_host = nullptr;

  half *sparseBcsrVal_dev = nullptr;
  int *sparseBcsrRowPtr_dev = nullptr;
  int *sparseBcsrColIdx_dev = nullptr;

  // New members for filtered dense blocks
  half *denseBcsrVal_host = nullptr;
  int *denseBcsrRowPtr_host = nullptr;
  int *denseBcsrColIdx_host = nullptr;

  half *denseBcsrVal_dev = nullptr;
  int *denseBcsrRowPtr_dev = nullptr;
  int *denseBcsrColIdx_dev = nullptr;

  // **** New members for sparse block info and sparse relative mapping ****
  // For consistency: a sparse block info array (all ones) for the filtered
  // sparse blocks
  int *sparseBlockInfo_host = nullptr;
  int *sparseBlockInfo_dev = nullptr;

  // A relative block index mapping for only the sparse blocks over the global
  // grid
  int *sparseRelativeBlockIndexMapping_host = nullptr;
  int *sparseRelativeBlockIndexMapping_dev = nullptr;

  size_t numberOfBlocks = 0;
  size_t nonzeroBlocks = 0;
  size_t sparseBlocks = 0;
  size_t denseBlocks = 0;
};
