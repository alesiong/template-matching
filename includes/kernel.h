#ifndef KERNEL_H
#define KERNEL_H
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements);
void RunKernel(void);

#endif
