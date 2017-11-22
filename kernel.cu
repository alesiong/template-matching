#include <cuda_runtime.h>
#include <stdio.h>
#include "includes/kernel.h"
#include <math.h>


__global__ void calcL1RowCumSum(const float *image, float *rowCumSum, int colNumberM) {
  float sum = 0;
  for (int i = 0; i < colNumberM; ++i) {
    sum += image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcL1SumTable(const float *rowCumSum, float *l1SumTable,
                           int rowNumberN, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < rowNumberN; i++) {
    sum += rowCumSum[i * colNumberM + blockIdx.x];
    l1SumTable[i * colNumberM + blockIdx.x] = sum;
  }
}

__global__ void calcL2RowCumSqrSum(const float *image, float *rowCumSum, int colNumberM) {
  float sum = 0;
  for (int i = 0; i < colNumberM; ++i) {
    sum += pow(image[threadIdx.x * colNumberM + i], 2);
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcL2SumTable(float *rowCumSum, float *l2SumTable,
                           int rowNumberN, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < rowNumberN; i++) {
    sum += rowCumSum[i * colNumberM + blockIdx.x];
    l1SumTable[i * colNumberM + blockIdx.x] = sum;
  }
}

__global__ void calcLxRowCumGradntSum(const float *image, float *rowCumSum, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < colNumberM; i++) {
    sum += threadIdx.x * image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcLxSumTable(const float *rowCumSum, float *lxSumTable,
                           int rowNumberN, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < rowNumberN; i++) {
    sum += rowCumSum[i * colNumberM + blockIdx.x];
    lxSumTable[i * colNumberM + blockIdx.x] = sum;
  }
}

__global__ void calcLyRowCumGradntSum(const float *image, float *rowCumSum, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < colNumberM; i++) {
    sum += i * image[threadIdx.x * colNumberM + i];
    rowCumSum[threadIdx.x * colNumberM + i] = sum;
  }
}

__global__ void calcLySumTable(const float *rowCumSum, float *lySumTable,
                           int rowNumberN, int colNumberM) {
  float sum = 0;
  for (size_t i = 0; i < rowNumberN; i++) {
    sum += rowCumSum[i * colNumberM + blockIdx.x];
    lySumTable[i * colNumberM + blockIdx.x] = sum;
  }
}

__global__ void allocateCudaMem(float *pointer, int size) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void **)&pointer, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void RunKernel(const float *I, const float *T, int M, int N, int K) {
  float *l1SumTable;
  float *l2SumTable;
  float *lxSumTable;
  float *lySumTable;

  allocateCudaMem(l1SumTable, M * N);
  allocateCudaMem(l2SumTable, M * N);
  allocateCudaMem(lxSumTable, M * N);
  allocateCudaMem(lySumTable, M * N);

  //calculate l1 sum table
  calcL1RowCumSum<<< 1, N>>>(I, l1SumTable, M);
  calcL1SumTable<<< 1, M>>>(l1SumTable, l1SumTable, N, M);



}

void RunKernel(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
}
