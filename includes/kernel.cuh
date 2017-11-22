#ifndef KERNEL_CUH
#define KERNEL_CUH

typedef struct SumTable_s {
  float* l1SumTable;
  float* l2SumTable;
  float* lxSumTable;
  float* lySumTable;
} SumTable;

typedef struct FeatureVector_s {
  float* meanVector;
  float* varianceVector;
  float* xGradientVector;
  float* yGradientVector;
} FeatureVector;

/*
 * Get the match of T in I, the output is put in x and y that representing the
 * middle point of the matching.
 */
void GetMatch(float* I, float* T, int Iw, int Ih, int Tw, int Th, int* x,
              int* y);

#endif
