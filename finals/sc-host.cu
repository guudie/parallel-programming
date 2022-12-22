#include "host.cuh"

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