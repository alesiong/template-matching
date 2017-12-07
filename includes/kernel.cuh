#ifndef KERNEL_CUH
#define KERNEL_CUH

/*
 * Get the match of T in I, the output is put in x and y that representing the
 * left bottom point
 */
void GetMatch(float* I, float* T, int Iw, int Ih, int Tw, int Th, int* x,
              int* y);

#endif
