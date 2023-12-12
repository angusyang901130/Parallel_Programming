#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int height, int maxIterations, int* result, size_t pitch) {
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

    char* row = (char*)result + thisY * pitch;
    int* elementAddress = (int*)(row + thisX * sizeof(int));
    *elementAddress = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // printf("width: %d, height: %d\n", resX, resY);Y
    // printf("upperX: %f, upperY, %f, lowerX: %f, lowerY: %f\n", upperX, upperY, lowerX, lowerY);
    // printf("stepX: %f, stepY: %f\n", stepX, stepY);

    int* deviceArray;
    size_t pitch;
    cudaMallocPitch((void**)&deviceArray, &pitch, resX * sizeof(int), resY);

    int nbytes = pitch * resY;

    int* hostArray;
    cudaHostAlloc((void**)&hostArray, nbytes, cudaHostAllocDefault);

    dim3 threadsPerBlock(16, 16);

    int blocks_x = resX % threadsPerBlock.x ? resX / threadsPerBlock.x + 1 : resX / threadsPerBlock.x;
    int blocks_y = resY % threadsPerBlock.y ? resY / threadsPerBlock.y + 1 : resY / threadsPerBlock.y;

    dim3 numBlocks(blocks_x, blocks_y);

    mandelKernel<<<numBlocks, threadsPerBlock>>> (lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, deviceArray, pitch);

    cudaMemcpy(hostArray, deviceArray, nbytes, cudaMemcpyDeviceToHost);
    
    for (int row = 0; row < resY; row++){

        char* hostRowAddress = (char*)hostArray + row * pitch;
        int* imgRowAddress = img + row * resX;

        memcpy(imgRowAddress, hostRowAddress, resX * sizeof(int));
    }

    cudaFreeHost(hostArray);
    cudaFree(deviceArray);

}
