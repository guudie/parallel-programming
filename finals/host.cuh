#ifndef HOST_CUH
#define HOST_CUH

#include "utils.cuh"

void seamCarvingCpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
                    int* xSobel, int* ySobel);

#endif