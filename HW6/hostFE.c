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
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, NULL);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_READ_WRITE, imageSize, NULL, NULL);

    // printf("Barrier 2\n");

    status = clEnqueueWriteBuffer(cmdQueue, inputBuffer, CL_TRUE, 0, imageSize, (void*)inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, filterBuffer, CL_TRUE, 0, filterSize, (void*)filter, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth);

    // printf("Barrier 3\n");

    size_t globalWorkSize[2] = {imageHeight, imageWidth};
    size_t localWorkSize[2] = {16, 16};
    clEnqueueNDRangeKernel(cmdQueue, kernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);

    clEnqueueReadBuffer(*context, outputImage, CL_TRUE, 0, imageSize, (void*)outputImage, NULL, NULL, NULL);

    // printf("Barrier 4\n");

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQueue);

}