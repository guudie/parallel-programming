#ifndef DONT_RUN_ON_HOST
    #define RUN_ON_HOST
#endif
#ifndef DONT_RUN_ON_DEVICE
    #define RUN_ON_DEVICE
#endif

#include "utils.cuh"
#ifdef RUN_ON_HOST
    #include "host.cuh"
#endif
#ifdef RUN_ON_DEVICE
    #include "device.cuh"
#endif

void seamCarving(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize1D=dim3(1024), dim3 blockSize2D=dim3(32, 32), bool useDevice=false) 
{
    GpuTimer timer;
	timer.Start();
    if(!useDevice) {
        #ifdef RUN_ON_HOST
            seamCarvingCpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel);
        #endif
    } else {
        #ifdef RUN_ON_DEVICE
            seamCarvingGpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel, blockSize1D, blockSize2D);
        #endif
    }
    timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
}

// inputs: filename, target width, block size (optional)
int main(int argc, char** argv) {
    if (argc > 3)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

    #ifdef RUN_ON_DEVICE
	    printDeviceInfo();
    #endif
    
    int width, height;
    int targetWidth = atoi(argv[2]);
    if(targetWidth < 1) {
        printf("Invalid target width\n");
        return EXIT_FAILURE;
    }
    
    uchar3* inPixels;
    readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

    int xSobel[9] = { 1, 0, -1,
                      2, 0, -2,
                      1, 0, -1 };
    int ySobel[9] = {  1,  2,  1,
                       0,  0,  0,
                      -1, -2, -1 };

    #ifdef RUN_ON_HOST
        uchar3* correctOut = (uchar3*)malloc(sizeof(uchar3) * targetWidth * height);
        seamCarving(inPixels, correctOut, width, height, targetWidth, xSobel, ySobel, false);
    #endif

    #ifdef RUN_ON_DEVICE
        dim3 blockSize1D(1024);
        dim3 blockSize2D(32, 32);
        uchar3* deviceOut = (uchar3*)malloc(sizeof(uchar3) * targetWidth * height);
        seamCarving(inPixels, deviceOut, width, height, targetWidth, xSobel, ySobel, blockSize1D, blockSize2D, true);
    #endif

    #if defined(RUN_ON_HOST) && defined(RUN_ON_DEVICE)
        printf("Error: %f", getErr(correctOut, deviceOut, targetWidth * height));
    #endif

    char* outFileNameBase = strtok(argv[1], ".");
    #ifdef RUN_ON_HOST
        char* tmpName = concatStr(outFileNameBase, argv[2]);
        writePnm(correctOut, targetWidth, height, concatStr(tmpName, (char*)"_host.pnm"));
        free(correctOut);
    #endif
    #ifdef RUN_ON_DEVICE
        char* tmpName = concatStr(outFileNameBase, argv[2]);
        writePnm(deviceOut, targetWidth, height, concatStr(tmpName, (char*)"_device.pnm"));
        free(deviceOut);
    #endif

    free(inPixels);
}

