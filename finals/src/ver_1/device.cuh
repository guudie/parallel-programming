#ifndef DEVICE_CUH
#define DEVICE_CUH

#include "utils.cuh"

__constant__ int d_xSobel[9];
__constant__ int d_ySobel[9];

__global__ void computeEnergyKernel(const uchar3* inPixels, int* energy, int width, int height);
__global__ void computeSeamsKernel(const int* energy, int2* dp, int width, int height);
__global__ void minReductionKernel(const int2* dp_lastRow, int width, int2* blockMin);
__global__ void carveSeamKernel(uchar3* inPixels, int* trace, int width, int height);
void seamCarvingGpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
                    int* xSobel, int* ySobel, dim3 blockSize1D=dim3(1024), dim3 blockSize2D=dim3(32, 32));

#endif