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

void seamCarving(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, bool useDevice=false) 
{
    GpuTimer timer;
	timer.Start();
    if(!useDevice) {
        seamCarvingCpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel);
    } else {

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

    writePnm(correctOut, targetWidth, height, "out_host.pnm");

    free(inPixels);
    free(correctOut);
}

