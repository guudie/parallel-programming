#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

template<typename T>
void swapPtr(T*& a, T*& b) {
    T* tmp = a;
    a = b;
    b = tmp;
}

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 * pixels, int width, int height, char * fileName);
float getErr(const uchar3* a, const uchar3* b, int n);
void printDeviceInfo();
char* concatStr(const char* s1, const char* s2);

#endif