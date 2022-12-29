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

// 1D grid call
__global__ void computeEnergySkipKernel(const uchar3* inPixels, int* energy, int width, int height, int* trace) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < height) {
        int toUpdateIdx[4] = {-1, -1, -1, -1};
        int curVal = trace[r];

        // update the two adjacent pixels of removed seam on row `r`: [r][trace[r] - 1] and [r][trace[r]]
        if(curVal > 0)
            toUpdateIdx[0] = r * width + curVal - 1;
        if(curVal < width)
            toUpdateIdx[1] = r * width + curVal;

        // update the top & bottom diagonal pixels
        int prevVal;
        int nextVal;
        if(r > 0 && (prevVal = trace[r-1]) != curVal) {
            if(prevVal > curVal && curVal > 0)
                toUpdateIdx[2] = (r - 1) * width + curVal - 1;
            if(prevVal < curVal && curVal < width)
                toUpdateIdx[2] = (r - 1) * width + curVal;
        }
        if(r < height - 1 && (nextVal = trace[r+1]) != curVal) {
            if(nextVal > curVal && curVal > 0)
                toUpdateIdx[3] = (r + 1) * width + curVal - 1;
            if(nextVal < curVal && curVal < width)
                toUpdateIdx[3] = (r + 1) * width + curVal;
        }

        for(int idx = 0; idx < 4; idx++) {
            if(toUpdateIdx[idx] == -1)
                continue;
            int rIdx = toUpdateIdx[idx] / width;
            int cIdx = toUpdateIdx[idx] % width;
            int x = 0, y = 0;
            for(int fri = 0; fri < 3; fri++) {
                for(int fci = 0; fci < 3; fci++) {
                    int f_r = rIdx - 1 + fri;
                    int f_c = cIdx - 1 + fci;
                    uchar3 val = (f_r >= 0 && f_r < height && f_c >= 0 && f_c < width) ? inPixels[f_r * width + f_c] : make_uchar3(0, 0, 0);
                    x += (val.x + val.y + val.z) / 3 * d_xSobel[fri * 3 + fci];
                    y += (val.x + val.y + val.z) / 3 * d_ySobel[fri * 3 + fci];
                }
            }
            atomicExch(&energy[toUpdateIdx[idx]], abs(x) + abs(y));
        }
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

// inPixels1 contains pixel info for the current iteration, inPixels2 will be calculated to hold info for the next
__global__ void carveSeamKernel(uchar3* inPixels1, uchar3* inPixels2, int* energy1, int* energy2, int* trace, int width, int height) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < height && c < width - 1) {
        inPixels2[r * (width - 1) + c] = inPixels1[r * width + c + (c >= trace[r])];
        energy2[r * (width - 1) + c] = energy1[r * width + c + (c >= trace[r])];
    }
}

void seamCarvingGpu(const uchar3* inPixels, uchar3* outPixels, int width, int height, int targetWidth,
        int* xSobel, int* ySobel, dim3 blockSize1D, dim3 blockSize2D)
{
    dim3 blockSizeEnergy = blockSize2D;
    dim3 blockSizeEnergySkip = blockSize1D;
    dim3 blockSizeSeams = blockSize1D;
    dim3 blockSizeReduction = blockSize1D;
    dim3 blockSizeCarve = blockSize2D;
    ///////////////////////////////////////////

    uchar3 *d_inPixels1, *d_inPixels2;
    int *d_energy1, *d_energy2, *d_trace;
    int2 *d_dp, *d_blockMin;

    int *trace;
    int2 *dp, *blockMin;

    size_t inSize = sizeof(uchar3) * width * height;
    size_t arrSize = sizeof(int) * width * height;

    CHECK(cudaMalloc(&d_inPixels1, inSize));
    CHECK(cudaMalloc(&d_inPixels2, inSize));

    CHECK(cudaMalloc(&d_energy1, arrSize));
    CHECK(cudaMalloc(&d_energy2, arrSize));
    CHECK(cudaMalloc(&d_trace, arrSize));
    CHECK(cudaMalloc(&d_dp, sizeof(int2) * width * height));
    CHECK(cudaMalloc(&d_blockMin, sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1)));

    dp = (int2*)malloc(sizeof(int2) * width * height);
    trace = (int*)malloc(sizeof(int) * height);
    blockMin = (int2*)malloc(sizeof(int2) * ((width - 1) / blockSizeReduction.x / 2 + 1));

    CHECK(cudaMemcpy(d_inPixels1, inPixels, inSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_xSobel, xSobel, sizeof(int) * 9));
    CHECK(cudaMemcpyToSymbol(d_ySobel, ySobel, sizeof(int) * 9));

    dim3 gridSizeEnergy((width - 1) / blockSizeEnergy.x + 1, (height - 1) / blockSizeEnergy.y + 1);
    dim3 gridSizeEnergySkip((height - 1) / blockSizeEnergySkip.x + 1);
    for(int curWidth = width; curWidth > targetWidth; curWidth--) {
        dim3 gridSizeSeams(1);
        dim3 gridSizeReduction((curWidth - 1) / blockSizeReduction.x / 2 + 1);
        dim3 gridSizeCarve((curWidth - 1) / blockSizeCarve.x + 1, (height - 1) / blockSizeCarve.y + 1);

        int smemEnergy = (blockSizeEnergy.x + 2) * (blockSizeEnergy.y + 2) * sizeof(uchar3);
        int smemSeams = 2 * curWidth * sizeof(int);
        int smemReduction = 2 * blockSizeReduction.x * sizeof(int2);

        // compute energy
        if(curWidth == width)
            computeEnergyKernel<<<gridSizeEnergy, blockSizeEnergy, smemEnergy>>>(d_inPixels1, d_energy1, curWidth, height);
        else
            computeEnergySkipKernel<<<gridSizeEnergySkip, blockSizeEnergySkip>>>(d_inPixels1, d_energy1, curWidth, height, d_trace);

        // dynamic programming
        computeSeamsKernel<<<gridSizeSeams, blockSizeSeams, smemSeams>>>(d_energy1, d_dp, curWidth, height);
        // reduction to find min
        minReductionKernel<<<gridSizeReduction, blockSizeReduction, smemReduction>>>(d_dp + (height - 1) * curWidth, curWidth, d_blockMin);

        CHECK(cudaMemcpy(blockMin, d_blockMin, sizeof(int2) * gridSizeReduction.x, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(dp, d_dp, sizeof(int2) * curWidth * height, cudaMemcpyDeviceToHost));

        int2 res = blockMin[0];
        for(int i = 1; i < gridSizeReduction.x; i++)
            if(res.x > blockMin[i].x)
                res = blockMin[i];
        trace[height - 1] = res.y;

        // tracing
        for(int r = height - 1; r > 0; r--) {
            trace[r - 1] = dp[r * curWidth + trace[r]].y;
        }

        CHECK(cudaMemcpy(d_trace, trace, sizeof(int) * height, cudaMemcpyHostToDevice));
        // remove seam
        carveSeamKernel<<<gridSizeCarve, blockSizeCarve>>>(d_inPixels1, d_inPixels2, d_energy1, d_energy2, d_trace, curWidth, height);

        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        swapPtr(d_inPixels1, d_inPixels2);
        int* tmp = d_energy1;
        d_energy1 = d_energy2;
        d_energy2 = tmp;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels1, sizeof(uchar3) * targetWidth * height, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels1));
    CHECK(cudaFree(d_inPixels2));
    CHECK(cudaFree(d_energy1));
    CHECK(cudaFree(d_energy2));
    CHECK(cudaFree(d_trace));
    CHECK(cudaFree(d_dp));
    CHECK(cudaFree(d_blockMin));
    free(trace);
    free(dp);
    free(blockMin);
}
