#ifndef UTILS_CUH
#define UTILS_CUH

typedef struct SumTable_s {
  float* l1SumTable;
  float* l2SumTable;
  float* lxSumTable;
  float* lySumTable;
} SumTable;

/*
 * Get the information of maximal threads supported per block and the number
 * of SP per SM
 */
void GetDeviceInfo(int* maxThreadsPerBlock, int* workingThreadsPerBlock);

/*
 * Helper function to allocate memory on device, report error if OOM
 */
void AllocateCudaMem(float** pointer, int size);

#endif
