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
  for (int i = 0; i < colNumberM; i++) {
    sum += i * image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcLyRowCumGradntSum(const float *image, float *rowCumSum,
                                      int colNumberM) {
  float sum = 0;
  for (int i = 0; i < colNumberM; i++) {
    sum += threadIdx.x * image[threadIdx.x * colNumberM + i];
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

__device__ float computeS(float *sumTable, int rowNumberN, int colNumberM,
                          int startX, int startY, int Kx, int Ky) {
  startX--;
  startY--;
  float S =
      sumTable[startX + Kx + (Ky + startY) * colNumberM] -
      (startX < 0 ? 0 : sumTable[startX + Kx + startY * colNumberM]) -
      (startY < 0 ? 0 : sumTable[startX + (Ky + startY) * colNumberM]) +
      (startX < 0 || startY < 0 ? 0 : sumTable[startX + startY * colNumberM]);
  return S;
}

// totally (M - K + 1) * (N - K + 1) threads
__global__ void calcVectorFeatures(float *templateFeatures, int rowNumberN,
                                   int colNumberM, float *l1SumTable,
                                   float *l2SumTable, float *lxSumTable,
                                   float *lySumTable, int Kx, int Ky,
                                   float *differences) {
  float meanVector;
  float varianceVector;
  float xGradientVector;
  float yGradientVector;
  int startX = blockIdx.x;
  int startY = threadIdx.x;

  float S1D =
      computeS(l1SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  float S2D =
      computeS(l2SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

  meanVector = S1D / (Kx * Ky);

  varianceVector = S2D / (Kx * Ky) - powf(meanVector, 2);

  float SxD =
      computeS(lxSumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

  xGradientVector = 4 * (SxD - (startX + Kx / 2.0) * S1D) / (Kx * Kx * Ky);

  float SyD =
      computeS(lySumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
  yGradientVector = 4 * (SyD - (startY + Ky / 2.0) * S1D) / (Ky * Ky * Kx);

  differences[startX + startY * gridDim.x] = norm4df(
      templateFeatures[0] - meanVector, templateFeatures[1] - varianceVector,
      templateFeatures[2] - xGradientVector,
      templateFeatures[3] - yGradientVector);
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

void Preprocess(const float *I, const float *T, int M, int N, int Kx, int Ky,
                SumTable *sumTable, float *featuresT) {
  float *l1SumTable;
  float *l2SumTable;
  float *lxSumTable;
  float *lySumTable;

  allocateCudaMem(&l1SumTable, sizeof(float) * M * N);
  allocateCudaMem(&l2SumTable, sizeof(float) * M * N);
  allocateCudaMem(&lxSumTable, sizeof(float) * M * N);
  allocateCudaMem(&lySumTable, sizeof(float) * M * N);

  float *dev_I;

  allocateCudaMem(&dev_I, sizeof(float) * M * N);

  cudaMemcpy(dev_I, I, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // Use streams to ensure the order
  cudaStream_t l1Stream, l2Stream, lxStream, lyStream;
  cudaStreamCreate(&l1Stream);
  cudaStreamCreate(&l2Stream);
  cudaStreamCreate(&lxStream);
  cudaStreamCreate(&lyStream);

  // calculate sum tables first by row
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

  for (int i = 0; i < Ky; i++) {
    for (int j = 0; j < Kx; j++) {
      featuresT[0] += T[i * Kx + j];
      featuresT[1] += T[i * Kx + j] * T[i * Kx + j];
      featuresT[2] += j * T[i * Kx + j];
      featuresT[3] += i * T[i * Kx + j];
    }
  }

  featuresT[0] /= (float)(Kx * Ky);
  featuresT[1] = featuresT[1] / (float)(Kx * Ky) - featuresT[0] * featuresT[0];
  //   4/K^3*(Sx(D)-x*S1(D)), where x = Kx/2
  // = 4/K^3*(f2-Kx/2*f0*Kx*Ky)
  // = 4/Kx^2Ky*f2-2*f0
  featuresT[2] = 4.0 / (Kx * Kx * Ky) * featuresT[2] - 2.0 * featuresT[0];
  featuresT[3] = 4.0 / (Ky * Kx * Ky) * featuresT[3] - 2.0 * featuresT[0];

  cudaDeviceSynchronize();
  sumTable->l1SumTable = l1SumTable;
  sumTable->l2SumTable = l2SumTable;
  sumTable->lxSumTable = lxSumTable;
  sumTable->lySumTable = lySumTable;
}

void getMinimum(float *target, int M, int N, int *x, int *y) {
  float minimum = *target;
  *x = 0;
  *y = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (target[i * M + j] < minimum) {
        minimum = target[i * M + j];
        *x = j;
        *y = i;
      }
    }
  }
}

void GetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x,
              int *y) {
  SumTable sumTable;
  float featuresT[4] = {0, 0, 0, 0};
  Preprocess(I, T, Iw, Ih, Tw, Th, &sumTable, featuresT);
  float *dev_difference;
  float *difference;
  float *dev_featuresT;
  difference = (float *)malloc(sizeof(float) * (Iw - Tw + 1) * (Ih - Th + 1));
  allocateCudaMem(&dev_featuresT, sizeof(float) * 4);
  allocateCudaMem(&dev_difference,
                  sizeof(float) * (Iw - Tw + 1) * (Ih - Th + 1));
  cudaMemcpy(dev_featuresT, featuresT, sizeof(float) * 4,
             cudaMemcpyHostToDevice);

  // calcVectorFeatures(float *templateFeatures, int rowNumberN, int colNumberM,
  //                    float *l1SumTable, float *l2SumTable, float *lxSumTable,
  //                    float *lySumTable, int Kx, int Ky, float *differences)

  calcVectorFeatures<<<Iw - Tw + 1, Ih - Th + 1>>>(
      dev_featuresT, Ih, Iw, sumTable.l1SumTable, sumTable.l2SumTable,
      sumTable.lxSumTable, sumTable.lySumTable, Tw, Th, dev_difference);
  cudaDeviceSynchronize();

  // reduceMinRow<<<(Iw - Tw + 1), (Iw - Tw + 1)>>>(dev_difference, );
  cudaMemcpy(difference, dev_difference,
             sizeof(float) * (Iw - Tw + 1) * (Ih - Th + 1),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // find the max, by kernel?
  getMinimum(difference, Iw - Tw + 1, Ih - Th + 1, x, y);
  cudaFree(sumTable.l1SumTable);
  cudaFree(sumTable.l2SumTable);
  cudaFree(sumTable.lxSumTable);
  cudaFree(sumTable.lySumTable);
  cudaFree(dev_difference);
  cudaFree(dev_featuresT);
  free(difference);
}
