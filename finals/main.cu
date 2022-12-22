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

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void seamCarvingCpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel) 
{
    uchar3* curIn = (uchar3*)malloc(sizeof(uchar3) * width * height);
    memcpy(curIn, inPixels, sizeof(uchar3) * width * height);

    int* energy = (int*)malloc(sizeof(int) * width * height);
    int* dp = (int*)malloc(sizeof(int) * width * height);
    int* trace = (int*)malloc(sizeof(int) * height);

    // loop while there are seams to carve
    for(int curWidth = width; curWidth > targetWidth; curWidth--) {
        // edge detection convolution
        if(curWidth == width) {
            for(int r = 0; r < height; r++) {
                for(int c = 0; c < curWidth; c++) {
                    int x = 0, y = 0;
                    for(int fri = 0; fri < 3; fri++) {
                        for(int fci = 0; fci < 3; fci++) {
                            int f_r = r - 3 / 2 + fri;
                            int f_c = c - 3 / 2 + fci;
                            uchar3 val = (f_r >= 0 && f_r < height && f_c >= 0 && f_c < curWidth) ? curIn[f_r * curWidth + f_c] : make_uchar3(0, 0, 0);
                            x += (val.x + val.y + val.z) / 3 * xSobel[fri * 3 + fci];
                            y += (val.x + val.y + val.z) / 3 * ySobel[fri * 3 + fci];
                        }
                    }
                    energy[r * curWidth + c] = abs(x) + abs(y);
                }
            }
        } else {
            // 0: left, 1: right, 2: top, 3: bottom;
            int* toUpdateIdx = (int*)malloc(sizeof(int) * 4);
            for(int r = 0; r < height; r++) {
                toUpdateIdx[0] = toUpdateIdx[1] = toUpdateIdx[2] = toUpdateIdx[3] = -1;
                // update the two adjacent pixels of removed seam on row `r`: [r][trace[r] - 1] and [r][trace[r]]
                if(trace[r] > 0)
                    toUpdateIdx[0] = r * curWidth + trace[r] - 1;
                if(trace[r] < curWidth)
                    toUpdateIdx[1] = r * curWidth + trace[r];

                // update the top & bottom diagonal pixels
                if(r > 0 && trace[r-1] != trace[r]) {
                    if(trace[r-1] > trace[r] && trace[r] > 0)
                        toUpdateIdx[2] = (r - 1) * curWidth + trace[r] - 1;
                    if(trace[r-1] < trace[r] && trace[r] < curWidth)
                        toUpdateIdx[2] = (r - 1) * curWidth + trace[r];
                }
                if(r < height - 1 && trace[r+1] != trace[r]) {
                    if(trace[r+1] > trace[r] && trace[r] > 0)
                        toUpdateIdx[3] = (r + 1) * curWidth + trace[r] - 1;
                    if(trace[r+1] < trace[r] && trace[r] < curWidth)
                        toUpdateIdx[3] = (r + 1) * curWidth + trace[r];
                }

                for(int idx = 0; idx < 4; idx++) {
                    if(toUpdateIdx[idx] == -1)
                        continue;
                    int rIdx = toUpdateIdx[idx] / curWidth;
                    int cIdx = toUpdateIdx[idx] % curWidth;
                    int x = 0, y = 0;
                    for(int fri = 0; fri < 3; fri++) {
                        for(int fci = 0; fci < 3; fci++) {
                            int f_r = rIdx - 3 / 2 + fri;
                            int f_c = cIdx - 3 / 2 + fci;
                            uchar3 val = (f_r >= 0 && f_r < height && f_c >= 0 && f_c < curWidth) ? curIn[f_r * curWidth + f_c] : make_uchar3(0, 0, 0);
                            x += (val.x + val.y + val.z) / 3 * xSobel[fri * 3 + fci];
                            y += (val.x + val.y + val.z) / 3 * ySobel[fri * 3 + fci];
                        }
                    }
                    energy[toUpdateIdx[idx]] = abs(x) + abs(y);
                }
            }
            free(toUpdateIdx);
        }

        // calculate seams
        for(int c = 0; c < curWidth; c++) {
            dp[c] = energy[c];
        }
        for(int r = 1; r < height; r++) {
            for(int c = 0; c < curWidth; c++) {
                int i = r * curWidth + c;
                dp[i] = dp[(r-1) * curWidth + c];
                if(c - 1 >= 0)
                    dp[i] = min(dp[i], dp[(r-1) * curWidth + c - 1]);
                if(c + 1 < curWidth)
                    dp[i] = min(dp[i], dp[(r-1) * curWidth + c + 1]);
                dp[i] += energy[i];
            }
        }

        // reduction on last row
        int res = dp[(height - 1) * curWidth];
        trace[height - 1] = 0;
        for(int c = 1; c < curWidth; c++) {
            if(res > dp[(height - 1) * curWidth + c]) {
                res = dp[(height - 1) * curWidth + c];
                trace[height - 1] = c;
            }
        }

        // tracing
        for(int r = height - 1; r > 0; r--) {
            for(int c_top = max(0, trace[r] - 1); c_top <= min(trace[r] + 1, curWidth - 1); c_top++) {
                if(dp[(r - 1) * curWidth + c_top] + energy[r * curWidth + trace[r]] == dp[r * curWidth + trace[r]]) {
                    trace[r-1] = c_top;
                    break;
                }
            }
        }

        // remove seam from image
        for(int r = 0; r < height; r++) {
            for(int c = 0; c < curWidth - 1; c++) {
                curIn[r * (curWidth - 1) + c] = curIn[r * curWidth + c + (c >= trace[r])];
                energy[r * (curWidth - 1) + c] = energy[r * curWidth + c + (c >= trace[r])];
            }
        }
    }

    memcpy(outPixels, curIn, sizeof(uchar3) * targetWidth * height);
    free(curIn);
    free(energy);
    free(dp);
    free(trace);
}

__constant__ int d_xSobel[9];
__constant__ int d_ySobel[9];

__global__ void computeEnergyKernel(const uchar3* inPixels, int* energy, int width, int height) {
    extern __shared__ uchar3 s_inPixels[];
    int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int dimXWithFilter = blockDim.x + 2;
    uchar3 zero_uchar3 = {0, 0, 0};
	s_inPixels[(threadIdx.y + 1) * dimXWithFilter + threadIdx.x + 1] = (r < height && c < width) ? inPixels[r * width + c] : zero_uchar3;
	if(threadIdx.y < 1) {
		// top apron
		int rt = r - 1;
		s_inPixels[threadIdx.y * dimXWithFilter + threadIdx.x + 1] = (rt >= 0 && c < width) ? inPixels[rt * width + c] : zero_uchar3;

		// bottom apron
		int rb = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
		s_inPixels[(threadIdx.y + blockDim.y + 1) * dimXWithFilter + threadIdx.x + 1] = (rb < height && c < width) ? inPixels[rb * width + c] : zero_uchar3;

		// left & right aprons
		int cl = blockIdx.x * blockDim.x - 1 + threadIdx.y;
		int cr = (blockIdx.x + 1) * blockDim.x + threadIdx.y;
        for(int idx = threadIdx.x; idx < blockDim.y + 2; idx += blockDim.x) {
            int tmpR = blockIdx.y * blockDim.y + idx - 1;
		    s_inPixels[idx * dimXWithFilter + threadIdx.y] = (cl >= 0 && 0 <= tmpR && tmpR < height) ? inPixels[tmpR * width + cl] : zero_uchar3;
		    s_inPixels[idx * dimXWithFilter + blockDim.x + 1 + threadIdx.y] = (cr < width && 0 <= tmpR && tmpR < height) ? inPixels[tmpR * width + cr] : zero_uchar3;
        }
	}
	__syncthreads();

    if(r < height && c < width) {
		int x = 0, y = 0;
		for(int f_r = 0; f_r < 3; f_r++) {
			for(int f_c = 0; f_c < 3; f_c++) {
				int ri = threadIdx.y + f_r;
				int ci = threadIdx.x + f_c;
				uchar3 val = s_inPixels[ri * dimXWithFilter + ci];
				x += (val.x + val.y + val.z) / 3 * d_xSobel[f_r * 3 + f_c];
				y += (val.x + val.y + val.z) / 3 * d_ySobel[f_r * 3 + f_c];
			}
		}
		energy[r * width + c] = abs(x) + abs(y);
	}
}

// called with 1 flat block
__global__ void computeSeamsKernel(const int* energy, int2* dp, int width, int height) {
    extern __shared__ int s_rows[]; // stores 2 consecutive rows for faster memory access
    for(int c = threadIdx.x; c < width; c += blockDim.x) {
        dp[c] = make_int2(energy[c], 0);
        s_rows[width + c] = energy[c];
    }
    __syncthreads();
    for(int r = 1; r < height; r++) {
        for(int c = threadIdx.x; c < width; c += blockDim.x) {
            int i = r * width + c;
            int2 res = make_int2(s_rows[(r & 1) * width + c], c);
            if(c - 1 >= 0)
                if(res.x >= s_rows[(r & 1) * width + c - 1])
                    res = make_int2(s_rows[(r & 1) * width + c - 1], c - 1);
            if(c + 1 < width)
                if(res.x > s_rows[(r & 1) * width + c + 1])
                    res = make_int2(s_rows[(r & 1) * width + c + 1], c + 1);
            res.x += energy[i];
            dp[i] = res;
            s_rows[(1 - (r & 1)) * width + c] = res.x;
        }
        __syncthreads();
    }
}

__global__ void minReductionKernel(const int2* dp_lastRow, int width, int2* blockMin) {
    extern __shared__ int2 s_data[];
    int c = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if(c < width) {
        s_data[threadIdx.x].x = dp_lastRow[c].x;
        s_data[threadIdx.x].y = c;
    }
    if(c + blockDim.x < width) {
        s_data[threadIdx.x + blockDim.x].x = dp_lastRow[c + blockDim.x].x;
        s_data[threadIdx.x + blockDim.x].y = c + blockDim.x;
    }
    __syncthreads();

    for(int stride = blockDim.x; stride > 0; stride /= 2) {
        if(threadIdx.x < stride)
            if(c + stride < width && s_data[threadIdx.x].x > s_data[threadIdx.x + stride].x)
                s_data[threadIdx.x] = s_data[threadIdx.x + stride];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        blockMin[blockIdx.x] = s_data[0];
}

// inPixels1 contains pixel info for the current iteration, inPixels2 will be calculated to hold info for the next
__global__ void carveSeamKernel(uchar3* inPixels1, uchar3* inPixels2, int* trace, int width, int height) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < height && c < width - 1)
        inPixels2[r * (width - 1) + c] = inPixels1[r * width + c + (c >= trace[r])];
}

// unoptimized version of carveSeamKernel(), no ptr swapping trick, ran with 1 flat block
// __global__ void carveSeamKernel_v0(uchar3* inPixels, int* trace, int width, int height) {
//     for(int r = blockIdx.x; r < height; r += gridDim.x) {
//         for(int offsetX = 0; offsetX < width - 1; offsetX += blockDim.x) { // using offset instead of column index to avoid synchronization issues
//             int c = offsetX + threadIdx.x;
//             uchar3 val;
//             if(c < width - 1)
//                 val = inPixels[r * width + c + (c >= trace[r])];
//             __syncthreads();
//             if(c < width - 1)
//                 inPixels[r * (width - 1) + c] = val;
//             __syncthreads();
//         }
//     }
// }

void swapPtr(uchar3*& a, uchar3*& b) {
    uchar3* tmp = a;
    a = b;
    b = tmp;
}

void seamCarvingGpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize=dim3(1))
{
    // temp values, used for debugging
    dim3 blockSizeEnergy(32, 32);
    dim3 blockSizeSeams(1024);
    dim3 blockSizeReduction(1024);
    dim3 blockSizeCarve(32, 32);
    ///////////////////////////////////////////

    uchar3 *d_inPixels1, *d_inPixels2;
    int *d_energy, *d_trace;
    int2 *d_dp, *d_blockMin;

    int *trace;
    int2 *dp, *blockMin;

    size_t inSize = sizeof(uchar3) * width * height;
    size_t arrSize = sizeof(int) * width * height;

    CHECK(cudaMalloc(&d_inPixels1, inSize));
    CHECK(cudaMalloc(&d_inPixels2, inSize));

    CHECK(cudaMalloc(&d_energy, arrSize));
    CHECK(cudaMalloc(&d_trace, arrSize));
    CHECK(cudaMalloc(&d_dp, sizeof(int2) * width * height));
    CHECK(cudaMalloc(&d_blockMin, sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1)));

    dp = (int2*)malloc(sizeof(int2) * width * height);
    trace = (int*)malloc(sizeof(int) * height);
    blockMin = (int2*)malloc(sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1));

    CHECK(cudaMemcpy(d_inPixels1, inPixels, inSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_xSobel, xSobel, sizeof(int) * 9));
    CHECK(cudaMemcpyToSymbol(d_ySobel, ySobel, sizeof(int) * 9));

    for(int curWidth = width; curWidth > targetWidth; curWidth--) {
        dim3 gridSizeEnergy((curWidth - 1) / blockSizeEnergy.x + 1, (height - 1) / blockSizeEnergy.y + 1);
        dim3 gridSizeSeams(1);
        dim3 gridSizeReduction((curWidth - 1) / blockSizeReduction.x / 2 + 1);
        dim3 gridSizeCarve((curWidth - 1) / blockSizeCarve.x + 1, (height - 1) / blockSizeCarve.y + 1);

        int smemEnergy = (blockSizeEnergy.x + 2) * (blockSizeEnergy.y + 2) * sizeof(uchar3);
        int smemSeams = 2 * curWidth * sizeof(int);
        int smemReduction = 2 * blockSizeReduction.x * sizeof(int2);

        // compute energy
        computeEnergyKernel<<<gridSizeEnergy, blockSizeEnergy, smemEnergy>>>(d_inPixels1, d_energy, curWidth, height);
        // dynamic programming
        computeSeamsKernel<<<gridSizeSeams, blockSizeSeams, smemSeams>>>(d_energy, d_dp, curWidth, height);
        // reduction to find min
        minReductionKernel<<<gridSizeReduction, blockSizeReduction, smemReduction>>>(d_dp + (height - 1) * curWidth, curWidth, d_blockMin);

        CHECK(cudaMemcpy(blockMin, d_blockMin, sizeof(int2) * gridSizeReduction.x, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dp, d_dp, sizeof(int2) * curWidth * height, cudaMemcpyDeviceToHost));

        int2 res = blockMin[0];
        for(int i = 1; i < gridSizeReduction.x; i++)
            if(res.x > blockMin[i].x)
                res = blockMin[i];
        trace[height - 1] = res.y;
        // int res = dp[(height - 1) * curWidth].x;
        // trace[height - 1] = 0;
        // for(int c = 0; c < curWidth; c++) {
        //     if(res > dp[(height - 1) * curWidth + c].x) {
        //         res = dp[(height - 1) * curWidth + c].x;
        //         trace[height - 1] = c;
        //     }
        // }

        // tracing
        for(int r = height - 1; r > 0; r--) {
            trace[r - 1] = dp[r * curWidth + trace[r]].y;
        }

        CHECK(cudaMemcpy(d_trace, trace, sizeof(int) * height, cudaMemcpyHostToDevice));
        // remove seam
        carveSeamKernel<<<gridSizeCarve, blockSizeCarve>>>(d_inPixels1, d_inPixels2, d_trace, curWidth, height);

        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        swapPtr(d_inPixels1, d_inPixels2);
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels1, sizeof(uchar3) * targetWidth * height, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels1));
    CHECK(cudaFree(d_inPixels2));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_trace));
    CHECK(cudaFree(d_dp));
    CHECK(cudaFree(d_blockMin));
    free(trace);
    free(dp);
    free(blockMin);
}

void seamCarving(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize=dim3(1), bool useDevice=false) 
{
    GpuTimer timer;
	timer.Start();
    if(!useDevice) {
        seamCarvingCpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel);
    } else {
        seamCarvingGpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel, blockSize);
    }
    timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
}

float getErr(const uchar3* a, const uchar3* b, int n) {
    float ans = 0;
    for(int i = 0; i < n; i++) {
        ans += abs(a[i].x - b[i].x) + abs(a[i].y - b[i].y) + abs(a[i].z - b[i].z);
    }
    return ans/n/3;
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
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}

// inputs: filename, target width, block size (optional)
int main(int argc, char** argv) {
    if (argc > 4)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();
    
    int width, height;
    int targetWidth = atoi(argv[2]);
    uchar3* inPixels;
    readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

    int xSobel[9] = { 1, 0, -1,
                      2, 0, -2,
                      1, 0, -1 };
    int ySobel[9] = {  1,  2,  1,
                       0,  0,  0,
                      -1, -2, -1 };

    uchar3* correctOut = (uchar3*)malloc(sizeof(uchar3) * targetWidth * height);
    seamCarving(inPixels, correctOut, width, height, targetWidth, xSobel, ySobel, false);

    uchar3* deviceOut = (uchar3*)malloc(sizeof(uchar3) * targetWidth * height);
    seamCarving(inPixels, deviceOut, width, height, targetWidth, xSobel, ySobel, dim3(1), true);

    printf("Error: %f", getErr(correctOut, deviceOut, targetWidth * height));

    writePnm(correctOut, targetWidth, height, "out_host.pnm");
    writePnm(deviceOut, targetWidth, height, "out_device.pnm");

    free(inPixels);
    free(correctOut);
    free(deviceOut);
}

