#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    // printf("Barrier 1\n");

    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(*context, *device, 0, &status);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_WRITE, imageSize * sizeof(float), NULL, NULL);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize* sizeof(float), NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_READ_WRITE, imageSize* sizeof(float), NULL, NULL);

    for (int i = 0; i < )

    // printf("Barrier 2\n");

    status = clEnqueueWriteBuffer(cmdQueue, inputBuffer, CL_TRUE, 0, imageSize* sizeof(float), (void*)inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, filterBuffer, CL_TRUE, 0, filterSize* sizeof(float), (void*)filter, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&filterBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&outputBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&filterWidth);

    // printf("Barrier 3\n");

    size_t globalWorkSize[2] = {imageWidth, imageHeight};
    size_t localWorkSize[2] = {8, 8};
    clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    clEnqueueReadBuffer(cmdQueue, outputBuffer, CL_TRUE, 0, imageSize * sizeof(float), (void*)outputImage, NULL, NULL, NULL);

    // for(int row = 0; row < imageHeight; row++){
    //     for(int col = 0; col < imageWidth; col++){
    //         printf("(%d, %d) = %d\n", row, col, outputImage[row * imageWidth + col]);
    //     }
    // }

    // printf("Barrier 4\n");

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQueue);

}