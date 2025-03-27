// logging_cuda.cuh
#ifndef LOGGING_CUDA_CUH
#define LOGGING_CUDA_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Debug macro
#define DEBUG 1 // Set to 0 to disable debug prints

#if DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_THREAD(thread_id, ...)                                     \
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == thread_id) {        \
    printf(__VA_ARGS__);                                                       \
  }

// New macro for executing code on specific thread when DEBUG is true
#define DEBUG_EXECUTE_ON_THREAD(thread_id, code)                               \
  do {                                                                         \
    __syncthreads();                                                           \
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == thread_id &&      \
        threadIdx.y == 0) {                                                    \
      code                                                                     \
    }                                                                          \
  } while (0);

// Helper function to print matrices
template <typename T>
__device__ static void debugPrintMatrix(const char *name, T *matrix, int rows,
                                        int cols) {
  printf("begin print matrix %s\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", __half2float(matrix[i * cols + j]));
    }
    printf("\n");
  }
  printf("end print matrix %s\n\n", name);
}

// Helper function to print bits
__device__ static void debugPrintBits(const char *name, char *data, int size) {
  printf("begin print %s as bits\n", name);
  for (int i = 0; i < size; i++) {
    for (int bit = 7; bit >= 0; bit--) {
      printf("%d", (data[i] >> bit) & 1);
    }
    printf(" ");
  }
  printf("\n");
  printf("end print %s as bits\n", name);
}

#else
#define DEBUG_PRINT(...)                                                       \
  do {                                                                         \
  } while (0);
#define DEBUG_PRINT_THREAD(thread_id, ...)                                     \
  do {                                                                         \
  } while (0);
#define DEBUG_EXECUTE_ON_THREAD(thread_id, code)                               \
  do {                                                                         \
  } while (0);

template <typename T>
__device__ static void debugPrintMatrix(const char *name, T *matrix, int rows,
                                        int cols) {}
__device__ static void debugPrintBits(const char *name, char *data, int size) {}
#endif // DEBUG

#endif // LOGGING_CUDA_CUH