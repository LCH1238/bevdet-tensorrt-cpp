#ifndef __COMMON_H__
#define __COMMON_H__

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

typedef unsigned char uchar;

// using MACRO to allocate memory inside CUDA kernel
#define NUM_3D_BOX_CORNERS 8
#define NUM_2D_BOX_CORNERS 4
#define NUM_THREADS 64 //  need to be changed when num_threads is changed

#define CHECK_CUDA(ans) { GPUAssert((ans), __FILE__, __LINE__); }

inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
};


#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))


#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define ALIGN_TO(X, Y)    (CEIL_DIVIDE(X, Y) * (Y))

#endif