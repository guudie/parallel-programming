#include "device.cuh"

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
    if(c < width)
        s_data[threadIdx.x] = make_int2(dp_lastRow[c].x, c);
    if(c + blockDim.x < width)
        s_data[threadIdx.x + blockDim.x] = make_int2(dp_lastRow[c + blockDim.x].x, c + blockDim.x);
    __syncthreads();

    for(int stride = blockDim.x; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            int2& a = s_data[threadIdx.x];
            int2 b = s_data[threadIdx.x + stride];
            if(c + stride < width && (a.x > b.x || (a.x == b.x && a.y > b.y)))
                a = b;
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        blockMin[blockIdx.x] = s_data[0];
}

// no ptr swapping trick, ran with 1 flat block
__global__ void carveSeamKernel(uchar3* inPixels, int* trace, int width, int height) {
    for(int r = blockIdx.x; r < height; r += gridDim.x) {
        for(int offsetX = 0; offsetX < width - 1; offsetX += blockDim.x) { // using offset instead of column index to avoid synchronization issues
            int c = offsetX + threadIdx.x;
            uchar3 val;
            if(c < width - 1)
                val = inPixels[r * width + c + (c >= trace[r])];
            __syncthreads();
            if(c < width - 1)
                inPixels[r * (width - 1) + c] = val;
            __syncthreads();
        }
    }
}

void seamCarvingGpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize1D, dim3 blockSize2D)
{
    // temp values, used for debugging
    dim3 blockSizeEnergy = blockSize2D;
    dim3 blockSizeSeams = blockSize1D;
    dim3 blockSizeReduction = blockSize1D;
    dim3 blockSizeCarve = blockSize1D;
    ///////////////////////////////////////////

    uchar3 *d_inPixels;
    int *d_energy, *d_trace;
    int2 *d_dp, *d_blockMin;

    int *trace;
    int2 *dp, *blockMin;

    size_t inSize = sizeof(uchar3) * width * height;
    size_t arrSize = sizeof(int) * width * height;

    CHECK(cudaMalloc(&d_inPixels, inSize));

    CHECK(cudaMalloc(&d_energy, arrSize));
    CHECK(cudaMalloc(&d_trace, arrSize));
    CHECK(cudaMalloc(&d_dp, sizeof(int2) * width * height));
    CHECK(cudaMalloc(&d_blockMin, sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1)));

    dp = (int2*)malloc(sizeof(int2) * width * height);
    trace = (int*)malloc(sizeof(int) * height);
    blockMin = (int2*)malloc(sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1));

    CHECK(cudaMemcpy(d_inPixels, inPixels, inSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_xSobel, xSobel, sizeof(int) * 9));
    CHECK(cudaMemcpyToSymbol(d_ySobel, ySobel, sizeof(int) * 9));

    for(int curWidth = width; curWidth > targetWidth; curWidth--) {
        dim3 gridSizeEnergy((curWidth - 1) / blockSizeEnergy.x + 1, (height - 1) / blockSizeEnergy.y + 1);
        dim3 gridSizeSeams(1);
        dim3 gridSizeReduction((curWidth - 1) / blockSizeReduction.x / 2 + 1);
        dim3 gridSizeCarve(1);

        int smemEnergy = (blockSizeEnergy.x + 2) * (blockSizeEnergy.y + 2) * sizeof(uchar3);
        int smemSeams = 2 * curWidth * sizeof(int);
        int smemReduction = 2 * blockSizeReduction.x * sizeof(int2);

        // compute energy
        computeEnergyKernel<<<gridSizeEnergy, blockSizeEnergy, smemEnergy>>>(d_inPixels, d_energy, curWidth, height);
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
        carveSeamKernel<<<gridSizeCarve, blockSizeCarve>>>(d_inPixels, d_trace, curWidth, height);

        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, sizeof(uchar3) * targetWidth * height, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_trace));
    CHECK(cudaFree(d_dp));
    CHECK(cudaFree(d_blockMin));
    free(trace);
    free(dp);
    free(blockMin);
}
