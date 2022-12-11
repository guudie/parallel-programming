// Last update: 16/12/2020
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

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

__global__ void maskBitKernel(uint32_t * in, int n, uint32_t * bits, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        bits[i] = (in[i] >> d) & 1;
}

__global__ void scanKernel(uint32_t * in, int n, uint32_t * out, volatile uint32_t * bSums)
{
    extern __shared__ uint32_t s_data[];
    __shared__ int bi;
    if(threadIdx.x == 0) {
        bi = atomicAdd(&bCount, 1);
        s_data[0] = 0;
    }
    __syncthreads();

    // copy to shared memory
    int i1 = bi * blockDim.x * 2 + threadIdx.x;
    int i2 = i1 + blockDim.x;
    if(threadIdx.x > 0)
        s_data[threadIdx.x] = i1 - 1 < n ? in[i1 - 1] : 0;
    s_data[threadIdx.x + blockDim.x] = i2 - 1 < n ? in[i2 - 1] : 0;
    __syncthreads();

    // reduction
    for(int stride = 1; stride < 2 * blockDim.x; stride *= 2) {
        int i = (threadIdx.x + 1) * stride * 2 - 1;
        if(i < 2 * blockDim.x)
            s_data[i] += s_data[i - stride];
        __syncthreads();
    }
    // post-reduction
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int i = (threadIdx.x + 1) * stride * 2 - 1 + stride;
        if(i < 2 * blockDim.x)
            s_data[i] += s_data[i - stride];
        __syncthreads();
    }

    // write sum of block to bSums
    if(threadIdx.x == 0) {
        int endIdx = (bi + 1) * blockDim.x * 2 - 1;
        bSums[bi] = s_data[2 * blockDim.x - 1] + (endIdx < n ? in[endIdx] : 0);
        
        if(bi > 0) {
            while(bCount1 < bi);
            bSums[bi] += bSums[bi - 1];
            __threadfence();
        }
        bCount1 += 1;
    }
    __syncthreads();

    // update block with previous block sum
    if(bi > 0) {
        s_data[threadIdx.x] += bSums[bi - 1];
        s_data[threadIdx.x + blockDim.x] += bSums[bi - 1];
    }

    // write to out
    if(i1 < n)
        out[i1] = s_data[threadIdx.x];
    if(i2 < n)
        out[i2] = s_data[threadIdx.x + blockDim.x];
}

__global__ void reorderKernel(uint32_t * in, uint32_t * bits, int n, uint32_t * out, uint32_t * nOnesBefore)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        int rank;
        if(bits[i]) {
            int nZeros = n - nOnesBefore[n-1] - bits[n-1];
            rank = nZeros + nOnesBefore[i];
        } else {
            rank = i - nOnesBefore[i];
        }
        out[rank] = in[i];
    }
}

void swapPtr(uint32_t *&a, uint32_t *&b)
{
    uint32_t *tmp = a;
    a = b;
    b = tmp;
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    uint32_t *d_in, *d_out, *bits, *nOnesBefore;
    volatile uint32_t *bSums;
    
    int gridSize = (n - 1) / blockSize + 1;
    int gridSizeScan = (n - 1) / blockSize / 2 + 1;

    size_t tmp = n * sizeof(uint32_t);
    CHECK(cudaMalloc(&d_in, tmp));
    CHECK(cudaMalloc(&d_out, tmp));
    CHECK(cudaMalloc(&bits, tmp));
    CHECK(cudaMalloc(&nOnesBefore, tmp));
    CHECK(cudaMalloc(&bSums, gridSizeScan * sizeof(uint32_t)));

    CHECK(cudaMemcpy(d_in, in, tmp, cudaMemcpyHostToDevice));

    int initVal = 0;
    for(int d = 0; d < 25; d++) {
        CHECK(cudaMemcpyToSymbol(bCount, &initVal, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &initVal, sizeof(int)));
        maskBitKernel<<<gridSize, blockSize>>>(d_in, n, bits, d);
        scanKernel<<<gridSizeScan, blockSize, 2 * blockSize * sizeof(uint32_t)>>>(bits, n, nOnesBefore, bSums);
        reorderKernel<<<gridSize, blockSize>>>(d_in, bits, n, d_out, nOnesBefore);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        swapPtr(d_in, d_out);
    }
    CHECK(cudaMemcpy(out, d_in, tmp, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(bits);
    cudaFree(nOnesBefore);
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    //printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    //printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
