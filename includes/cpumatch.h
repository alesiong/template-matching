#include <stdio.h>
#include <stdlib.h>

#define L1Func(I, x, y) (I)
#define L2Func(I, x, y) (powf(I, 2))
#define LxFunc(I, x, y) (x * I)
#define LyFunc(I, x, y) (y * I)

float norm4df(float a, float b,float c,float d){
  float n = powf(a, 2) + powf(b, 2) + powf(c, 2) + powf(d, 2);
  return n;
}

float computeS(float *sumTable, int rowNumberN, int colNumberM,
              int startX, int startY, int Kx, int Ky) {
    startX--;
    startY--;
    float S =
        sumTable[startX + Kx + (Ky + startY) * colNumberM] -
        (startX < 0 ? 0 : sumTable[startX + (Ky + startY) * colNumberM]) -
        (startY < 0 ? 0 : sumTable[startX + Kx + startY * colNumberM]) +
        (startX < 0 || startY < 0 ? 0 : sumTable[startX + startY * colNumberM]);
    return S;
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

void CalSumTable(float *I, int Iw, int Ih, int Tw, int Th ,
  float *l1SumTable, float *l2SumTable, float *lxSumTable, float *lySumTable){
    float suml1 =0;
    float suml2 =0;
    float sumlx =0;
    float sumly =0;
  for(int i =0; i<Ih; i++){
    suml1 =0;
    suml2 =0;
    sumlx =0;
    sumly =0;
    for (int j =0; j<Iw; j++){
      suml1 += L1Func(I[i*Iw+j],j,i);
      suml2 += L2Func(I[i*Iw+j],j,i);
      sumlx += LxFunc(I[i*Iw+j],j,i);
      sumly += LyFunc(I[i*Iw+j],j,i);
      l1SumTable[i*Iw+j] = suml1;
      l2SumTable[i*Iw+j] = suml2;
      lxSumTable[i*Iw+j] = sumlx;
      lySumTable[i*Iw+j] = sumly;
    }
  }
  for(int i =0; i<Iw; i++){
    for (int j =1; j<Ih; j++){
      l1SumTable[j*Iw+i] += l1SumTable[(j-1)*Iw+i];
      l2SumTable[j*Iw+i] += l2SumTable[(j-1)*Iw+i];
      lxSumTable[j*Iw+i] += lxSumTable[(j-1)*Iw+i];
      lySumTable[j*Iw+i] += lySumTable[(j-1)*Iw+i];
    }
      }
}


void CPUGetMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, int *x,
              int *y){

    float* l1SumTable;
    float* l2SumTable;
    float* lxSumTable;
    float* lySumTable;
    float *differences;
    size_t sumtablesize = sizeof(float) * Iw * Ih;
    size_t difference_size = sizeof(float) * (Iw - Tw + 1) * (Ih - Th + 1);
    differences = (float *)malloc(difference_size);
    l1SumTable = (float *)malloc(sumtablesize);
    l2SumTable = (float *)malloc(sumtablesize);
    lxSumTable = (float *)malloc(sumtablesize);
    lySumTable = (float *)malloc(sumtablesize);

    CalSumTable(I,Iw,Ih,Tw,Th,
      l1SumTable,l2SumTable,lxSumTable,lySumTable);


      float featuresT[4] = {0, 0, 0, 0};
      for (int i = 0; i < Th; i++) {
        for (int j = 0; j < Tw; j++) {
          featuresT[0] += T[i * Tw + j];
          featuresT[1] += T[i * Tw + j] * T[i * Tw + j];
          featuresT[2] += j * T[i * Tw + j];
          featuresT[3] += i * T[i * Tw + j];
        }
      }

      featuresT[0] /= (float)(Tw * Th);
      featuresT[1] = featuresT[1] / (float)(Tw * Th) - featuresT[0] * featuresT[0];
      featuresT[2] = 4.0 / (Tw * Tw * Th) * featuresT[2] - 2.0 * featuresT[0];
      featuresT[3] = 4.0 / (Th * Tw * Th) * featuresT[3] - 2.0 * featuresT[0];

for (int i=0; i<(Iw - Tw +1); i++){
  for (int j=0; j<(Ih - Th +1); j++){
      float S1D = computeS(l1SumTable, Ih, Iw, i, j, Tw, Th);
      float S2D = computeS(l2SumTable, Ih, Iw, i, j, Tw, Th);
      float SxD = computeS(lxSumTable, Ih, Iw, i, j, Tw, Th);
      float SyD = computeS(lySumTable, Ih, Iw, i, j, Tw, Th);

      float meanVector = S1D / (Tw * Th);
      float varianceVector = S2D / (Tw * Th) - powf(meanVector, 2);
      float xGradientVector = 4 * (SxD - (i + Tw / 2.0) * S1D) / (Tw * Tw * Th);
      float yGradientVector = 4 * (SyD - (j + Th / 2.0) * S1D) / (Th * Th * Tw);

      differences[i + j *(Iw - Tw+1)] = norm4df(
          featuresT[0] - meanVector, featuresT[1] - varianceVector,
          featuresT[2] - xGradientVector,
          featuresT[3] - yGradientVector);
        }
        }
        getMinimum(differences, Iw - Tw + 1 , Ih - Th + 1 , x, y);

}
