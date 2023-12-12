#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int height, int maxIterations, int* result) {
    // To avoid error caused by the floating number, use the following pseudo code
    //

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if(thisX >= width || thisY >= height){
        return;
    }
        
    
    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;

    float z_re = c_re;
    float z_im = c_im;
    int i;

    for (i = 0; i < maxIterations; ++i){

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;

        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    result[thisY * width + thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int N = resX * resY;

    // printf("width: %d, height: %d\n", resX, resY);Y
    // printf("upperX: %f, upperY, %f, lowerX: %f, lowerY: %f\n", upperX, upperY, lowerX, lowerY);
    // printf("stepX: %f, stepY: %f\n", stepX, stepY);

    int* hostArray = (int*)malloc(N * sizeof(int));

    int* deviceArray;
    cudaMalloc(&deviceArray, N * sizeof(int));

    dim3 threadsPerBlock(8, 4);

    int blocks_x = resX % threadsPerBlock.x ? resX / threadsPerBlock.x + 1 : resX / threadsPerBlock.x;
    int blocks_y = resY % threadsPerBlock.y ? resY / threadsPerBlock.y + 1 : resY / threadsPerBlock.y;

    dim3 numBlocks(blocks_x, blocks_y);

    mandelKernel<<<numBlocks, threadsPerBlock>>> (lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, deviceArray);

    cudaMemcpy(hostArray, deviceArray, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    memcpy(img, hostArray, N * sizeof(int));

    free(hostArray);
    cudaFree(deviceArray);

}
