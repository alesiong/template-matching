#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "includes/bmp_util.h"
#include "includes/kernel.cuh"

// TO DO: WRITE KERNELS HERE

int main() {
  // Just an example here - you are free to modify them
  int I_width, I_height, T_width, T_height;
  float *I, *T;
  int x1, y1, x2, y2;

  // set the file location of I, T, and Output
  char I_path[] = "lena.bmp";
  char T_path[] = "lena_t.bmp";
  char out_path[] = "output.bmp";

  I = ReadBMP(I_path, &I_width, &I_height);
  T = ReadBMP(T_path, &T_width, &T_height);

  if (I == 0 || T == 0) {
    exit(1);
  }

  int x, y;

  GetMatch(I, T, I_width, I_height, T_width, T_height, &x, &y);
  x1 = x - T_width / 2;
  x2 = x + T_width / 2;
  y1 = y - T_width / 2;
  y2 = y + T_width / 2;

  MarkAndSave(I_path, x1, y1, x2, y2, out_path);

  free(I);
  free(T);
  return 0;
}
