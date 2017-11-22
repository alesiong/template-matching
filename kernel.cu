#include <cuda_runtime.h>
#include <stdio.h>
#include "includes/kernel.cuh"

__global__ void calcL1RowCumSum(const float *image, float *rowCumSum,
                                int colNumberM) {
  float sum = 0;
  for (int i = 0; i < colNumberM; ++i) {
    sum += image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcL2RowCumSqrSum(const float *image, float *rowCumSum,
                                   int colNumberM) {
  float sum = 0;
  for (int i = 0; i < colNumberM; ++i) {
    sum += powf(image[threadIdx.x * colNumberM + i], 2);
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcLxRowCumGradntSum(const float *image, float *rowCumSum,
                                      int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < colNumberM; i++) {
    sum += threadIdx.x * image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcLyRowCumGradntSum(const float *image, float *rowCumSum,
                                      int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < colNumberM; i++) {
    sum += i * image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcSumTable(const float *rowCumSum, float *SumTable,
                             int rowNumberN, int colNumberM) {
  for (int i = 1; i < rowNumberN; i++) {
    SumTable[i * colNumberM + blockIdx.x] +=
        rowCumSum[(i - 1) * colNumberM + blockIdx.x];
  }
}

//total (M - K + 1) * (N - K + 1) threads
//rowNumber is (N - K + 1), colNumberM is (M - K + 1)
__global__ void calcVectorFeatures(float *vectorFeatures, int rowNumberN,
                                   int colNumberM, float *l1SumTable,
                                   float *l2SumTable, float *lxSumTable,
                                   float *lySumTable, int K) {
  int start = threadIdx.x;
  float S1D = l1SumTable[start + K * colNumberM + K] - l1SumTable[start + K] \
              - l1SumTable[start + k * colNumberM] + l1SumTable[start];

  vectorFeatures[threadIdx.x * 4 + 0] = S1D / pow(K, 2);

  vectorFeatures[threadIdx.x * 4 + 1] = (l2SumTable[start + K * colNumberM + K] \
                                         - l2SumTable[start + K] \
                                         - l2SumTable[start + k * colNumberM] \
                                         + l2SumTable[start]) / pow(K, 2) \
                                  - pow(vectorFeatures[threadIdx.x * 4 + 0], 2);
  SxD = lxSumTable[start + K * colNumberM + K] - lxSumTable[start + K] \
        - lxSumTable[start + k * colNumberM] + lxSumTable[start];
  vectorFeatures[threadIdx.x * 4 + 2] = 4 * (SxD - (start / colNumberM + float(K/2))
                                        * S1D) / pow(K, 3);

  SyD = lySumTable[start + K * colNumberM + K] - lySumTable[start + K] \
        - lySumTable[start + k * colNumberM] + lySumTable[start];
  vectorFeatures[threadIdx.x * 4 + 3] = 4 * (SyD - (start % colNumberM + float(K/2))
                                        * S1D) / pow(K, 3);

}

void allocateCudaMem(float **pointer, int size) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void **)pointer, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void Preprocess(const float *I, const float *T, int M, int N, int K) {
  float *l1SumTable;
  float *l2SumTable;
  float *lxSumTable;
  float *lySumTable;

  allocateCudaMem(&l1SumTable, M * N);
  allocateCudaMem(&l2SumTable, M * N);
  allocateCudaMem(&lxSumTable, M * N);
  allocateCudaMem(&lySumTable, M * N);

  float *dev_I;
  float *dev_T;
  // TODO: copy I and T to device

  cudaStream_t l1Stream, l2Stream, lxStream, lyStream;
  cudaStreamCreate(&l1Stream);
  cudaStreamCreate(&l2Stream);
  cudaStreamCreate(&lxStream);
  cudaStreamCreate(&lyStream);

  // calculate l1 sum table
  calcL1RowCumSum<<<1, N, 0, l1Stream>>>(I, l1SumTable, M);
  calcL2RowCumSqrSum<<<1, N, 0, l2Stream>>>(I, l2SumTable, M);
  calcLxRowCumGradntSum<<<1, N, 0, lxStream>>>(I, lxSumTable, M);
  calcLyRowCumGradntSum<<<1, N, 0, lyStream>>>(I, lySumTable, M);

  calcSumTable<<<1, M>>>(l1SumTable, l1SumTable, N, M);

  cudaStreamDestroy(l1Stream);
  cudaStreamDestroy(l2Stream);
  cudaStreamDestroy(lxStream);
  cudaStreamDestroy(lyStream);
}

void GetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x,
              int *y) {
  Preprocess(I, T, Iw, Ih, 0);
  *x = 100;
  *y = 100;
}
