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
    SumTable[i * colNumberM + threadIdx.x] +=
        rowCumSum[(i - 1) * colNumberM + threadIdx.x];
  }
}

__global__ float computeS(float *sumTable, int rowNumberN, int colNumberM,
                          int startX, int startY, int Kx, int Ky) {
  startX--;
  startY--;
  float S = sumTable[startX + Kx + (Ky + startY) * colNumberM]
            - startX<0?0: sumTable[startX + Kx + startY * colNumberM]
            - startY<0?0: sumTable[startX + (Ky + startY) * colNumberM]
            + (startX<0||startY<0)?0: sumTable[startX + startY * colNumberM];
  return S;
}

//totally (M - K + 1) * (N - K + 1) threads
__global__ void calcVectorFeatures(float *templateFeatures,
                                   int rowNumberN, int colNumberM, float *l1SumTable,
                                   float *l2SumTable, float *lxSumTable,
                                   float *lySumTable, int Kx, int Ky, float *differences) {
  FeatureVector featureVectors;
  int startX = blockIdx.x;
  int startY = threadIdx.x;

  float S1D = computeS(l1SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  float S2D = computeS(l2SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

  featureVectors.meanVector[startX + startY * (gridDim.x)] = S1D / (Kx * Ky);

  float V1D = featureVectors->meanVector[startX + startY * (colNumberM - Kx + 1)];
  featureVectors.varianceVector[startX + startY * (gridDim.x)] =
      S2D / (Kx * Ky) - pow(V1D, 2);

  float SxD = computeS(lxSumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  featureVectors.xGradientVector[startX + startY * (gridDim.x)] =
      4 * (SxD - (startX + Kx/2.0) * S1D) / (Kx * Kx * Ky);

  float SyD = computeS(lySumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  featureVectors.yGradientVector[startX + startY * (gridDim.x)] =
      4 * (SyD - (startY + Ky/2.0) * S1D) / (Ky * KY * Kx);

  //TO DO: calculate differences
  


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

void Preprocess(const float *I, const float *T, int M, int N, int K,
                SumTable *sumTable) {
  float *l1SumTable;
  float *l2SumTable;
  float *lxSumTable;
  float *lySumTable;

  allocateCudaMem(&l1SumTable, sizeof(float) * M * N);
  allocateCudaMem(&l2SumTable, sizeof(float) * M * N);
  allocateCudaMem(&lxSumTable, sizeof(float) * M * N);
  allocateCudaMem(&lySumTable, sizeof(float) * M * N);

  float *dev_I;
  float *dev_T;

  allocateCudaMem(&dev_I, sizeof(float) * M * N);

  cudaMemcpy(dev_I, I, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // TODO: copy T to device

  cudaStream_t l1Stream, l2Stream, lxStream, lyStream;
  cudaStreamCreate(&l1Stream);
  cudaStreamCreate(&l2Stream);
  cudaStreamCreate(&lxStream);
  cudaStreamCreate(&lyStream);

  // calculate l1 sum table
  calcL1RowCumSum<<<1, N, 0, l1Stream>>>(dev_I, l1SumTable, M);
  calcL2RowCumSqrSum<<<1, N, 0, l2Stream>>>(dev_I, l2SumTable, M);
  calcLxRowCumGradntSum<<<1, N, 0, lxStream>>>(dev_I, lxSumTable, M);
  calcLyRowCumGradntSum<<<1, N, 0, lyStream>>>(dev_I, lySumTable, M);

  calcSumTable<<<1, M, 0, l1Stream>>>(l1SumTable, l1SumTable, N, M);
  calcSumTable<<<1, M, 0, l2Stream>>>(l2SumTable, l2SumTable, N, M);
  calcSumTable<<<1, M, 0, lxStream>>>(lxSumTable, lxSumTable, N, M);
  calcSumTable<<<1, M, 0, lyStream>>>(lySumTable, lySumTable, N, M);

  cudaStreamDestroy(l1Stream);
  cudaStreamDestroy(l2Stream);
  cudaStreamDestroy(lxStream);
  cudaStreamDestroy(lyStream);

  cudaDeviceSynchronize();
  sumTable->l1SumTable = l1SumTable;
  sumTable->l2SumTable = l2SumTable;
  sumTable->lxSumTable = lxSumTable;
  sumTable->lySumTable = lySumTable;
}

void GetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x,
              int *y) {
  SumTable sumTable;
  Preprocess(I, T, Iw, Ih, 0, &sumTable);
  *x = 100;
  *y = 100;
}
