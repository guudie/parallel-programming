#include "utils.cuh"
#include "host.cuh"
#include "device.cuh"

void seamCarving(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize1D=dim3(1024), dim3 blockSize2D=dim3(32, 32), bool useDevice=false) 
{
    GpuTimer timer;
	timer.Start();
    if(!useDevice) {
        seamCarvingCpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel);
    } else {
        seamCarvingGpu(inPixels, outPixels, width, height, targetWidth, xSobel, ySobel, blockSize1D, blockSize2D);
    }
    timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
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

    dim3 blockSize1D(1024);
    dim3 blockSize2D(32, 32);
    uchar3* deviceOut = (uchar3*)malloc(sizeof(uchar3) * targetWidth * height);
    seamCarving(inPixels, deviceOut, width, height, targetWidth, xSobel, ySobel, blockSize1D, blockSize2D, true);

    printf("Error: %f", getErr(correctOut, deviceOut, targetWidth * height));

    writePnm(correctOut, targetWidth, height, "out_host.pnm");
    writePnm(deviceOut, targetWidth, height, "out_device.pnm");

    free(inPixels);
    free(correctOut);
    free(deviceOut);
}

