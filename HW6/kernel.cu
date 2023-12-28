#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "CycleTimer.h"

typedef unsigned char uchar;

__global__ void convolution(float* inputImage, float* filter, float* outputImage, int imageHeight, int imageWidth, int filterWidth)
{
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;  // row

    int halffilterSize = filterWidth / 2;
    
    int row_min = -halffilterSize < -imageY ? -imageY : -halffilterSize;
    int row_max = halffilterSize >= imageHeight - imageY ? imageHeight - imageY: halffilterSize;
    int col_min = -halffilterSize < -imageX ? -imageX : -halffilterSize;
    int col_max = halffilterSize >= imageWidth - imageX ? imageWidth - imageX: halffilterSize;
    
    float sum = 0.0f;

    for (int i = row_min; i <= row_max; i++)
    {
        for (int j = col_min; j <= col_max; j++)
        {
            float input_val = inputImage[(imageY + i) * imageWidth + imageX + j];
            float filter_val = filter[(i + halffilterSize) * filterWidth + j + halffilterSize];

            if (input_val != 0 && filter_val != 0)
            {
                sum += input_val * filter_val;
            }

            
        }
    }
    
    outputImage[imageY * imageWidth + imageX] = sum;
}

void serialConv(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum;
    int i, j, k, l;

    for (i = 0; i < imageHeight; i++)
    {
        // Iterate over the columns of the source image
        for (j = 0; j < imageWidth; j++)
        {
            sum = 0; // Reset sum for new source pixel
            // Apply the filter to the neighborhood
            for (k = -halffilterSize; k <= halffilterSize; k++)
            {
                for (l = -halffilterSize; l <= halffilterSize; l++)
                {
                    if (i + k >= 0 && i + k < imageHeight &&
                        j + l >= 0 && j + l < imageWidth)
                    {
                        sum += inputImage[(i + k) * imageWidth + j + l] *
                               filter[(k + halffilterSize) * filterWidth +
                                      l + halffilterSize];
                    }
                }
            }
            outputImage[i * imageWidth + j] = sum;
        }
    }
}

int compare(const void *a, const void *b)
{
   double *x = (double *)a;
   double *y = (double *)b;
   if (*x < *y)
      return -1;
   else if (*x > *y)
      return 1;
   return 0;
}

float *readFilter(const char *filename, int *filterWidth)
{
    printf("Reading filter data from %s\n", filename);

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        printf("Could not open filter file\n");
        exit(-1);
    }

    fscanf(fp, "%d", filterWidth);

    float *filter = (float *)malloc(*filterWidth * *filterWidth * sizeof(int));

    float tmp;
    for (int i = 0; i < *filterWidth * *filterWidth; i++)
    {
        fscanf(fp, "%f", &tmp);
        filter[i] = tmp;
    }

    printf("Filter width: %d\n", *filterWidth);

    fclose(fp);
    return filter;
}

void storeImage(float *imageOut, const char *filename, int rows, int cols,
                const char *refFilename)
{

    FILE *ifp, *ofp;
    unsigned char tmp;
    int offset;
    unsigned char *buffer;
    int i, j;

    int bytes;

    int height, width;

    ifp = fopen(refFilename, "rb");
    if (ifp == NULL)
    {
        perror(filename);
        exit(-1);
    }

    fseek(ifp, 10, SEEK_SET);
    fread(&offset, 4, 1, ifp);

    fseek(ifp, 18, SEEK_SET);
    fread(&width, 4, 1, ifp);
    fread(&height, 4, 1, ifp);

    fseek(ifp, 0, SEEK_SET);

    buffer = (unsigned char *)malloc(offset);
    if (buffer == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    fread(buffer, 1, offset, ifp);

    printf("Writing output image to %s\n", filename);
    ofp = fopen(filename, "wb");
    if (ofp == NULL)
    {
        perror("opening output file");
        exit(-1);
    }
    bytes = fwrite(buffer, 1, offset, ofp);
    if (bytes != offset)
    {
        printf("error writing header!\n");
        exit(-1);
    }

    // NOTE bmp formats store data in reverse raster order (see comment in
    // readImage function), so we need to flip it upside down here.
    int mod = width % 4;
    if (mod != 0)
    {
        mod = 4 - mod;
    }
    //   printf("mod = %d\n", mod);
    for (i = height - 1; i >= 0; i--)
    {
        for (j = 0; j < width; j++)
        {
            tmp = (unsigned char)imageOut[i * cols + j];
            fwrite(&tmp, sizeof(char), 1, ofp);
        }
        // In bmp format, rows must be a multiple of 4-bytes.
        // So if we're not at a multiple of 4, add junk padding.
        for (j = 0; j < mod; j++)
        {
            fwrite(&tmp, sizeof(char), 1, ofp);
        }
    }

    fclose(ofp);
    fclose(ifp);

    free(buffer);
    }

    /*
    * Read bmp image and convert to byte array. Also output the width and height
    */
    float *readImage(const char *filename, int *widthOut, int *heightOut)
    {

    uchar *imageData;

    int height, width;
    uchar tmp;
    int offset;
    int i, j;

    printf("Reading input image from %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        perror(filename);
        exit(-1);
    }

    fseek(fp, 10, SEEK_SET);
    fread(&offset, 4, 1, fp);

    fseek(fp, 18, SEEK_SET);
    fread(&width, 4, 1, fp);
    fread(&height, 4, 1, fp);

    printf("width = %d\n", width);
    printf("height = %d\n", height);

    *widthOut = width;
    *heightOut = height;

    imageData = (uchar *)malloc(width * height);
    if (imageData == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    fseek(fp, offset, SEEK_SET);
    fflush(NULL);

    int mod = width % 4;
    if (mod != 0)
    {
        mod = 4 - mod;
    }

    // NOTE bitmaps are stored in upside-down raster order.  So we begin
    // reading from the bottom left pixel, then going from left-to-right,
    // read from the bottom to the top of the image.  For image analysis,
    // we want the image to be right-side up, so we'll modify it here.

    // First we read the image in upside-down

    // Read in the actual image
    for (i = 0; i < height; i++)
    {

        // add actual data to the image
        for (j = 0; j < width; j++)
        {
            fread(&tmp, sizeof(char), 1, fp);
            imageData[i * width + j] = tmp;
        }
        // For the bmp format, each row has to be a multiple of 4,
        // so I need to read in the junk data and throw it away
        for (j = 0; j < mod; j++)
        {
            fread(&tmp, sizeof(char), 1, fp);
        }
    }

    // Then we flip it over
    int flipRow;
    for (i = 0; i < height / 2; i++)
    {
        flipRow = height - (i + 1);
        for (j = 0; j < width; j++)
        {
            tmp = imageData[i * width + j];
            imageData[i * width + j] = imageData[flipRow * width + j];
            imageData[flipRow * width + j] = tmp;
        }
    }

    fclose(fp);

    // Input image on the host
    float *floatImage = NULL;
    floatImage = (float *)malloc(sizeof(float) * width * height);
    if (floatImage == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    // Convert the BMP image to float (not required)
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            floatImage[i * width + j] = (float)imageData[i * width + j];
        }
    }

    free(imageData);
    return floatImage;
    }


void hostFE(float* inputImage, float* filter, float* outputImage, int imageHeight, int imageWidth, int filterWidth)
{
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    float* deviceInputImage;
    float* deviceFilter;
    float* deviceOutputImage;

    cudaMalloc(&deviceInputImage, imageSize * sizeof(float));
    cudaMalloc(&deviceFilter, filterSize * sizeof(float));
    cudaMalloc(&deviceOutputImage, imageSize * sizeof(float));

    cudaMemcpy(deviceInputImage, inputImage, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);

    int block_x = imageWidth / threadsPerBlock.x; 
    int block_y = imageHeight / threadsPerBlock.y;

    dim3 numBlocks(block_x, block_y);

    convolution<<<numBlocks, threadsPerBlock>>>(deviceInputImage, deviceFilter, deviceOutputImage, imageHeight, imageWidth, filterWidth);

    cudaMemcpy(outputImage, deviceOutputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInputImage);
    cudaFree(deviceFilter);
    cudaFree(deviceOutputImage);
}

int main(){

    int i, j;

   // Rows and columns in the input image
    int imageHeight;
    int imageWidth;

    double start_time, end_time;

    char *inputFile = "input.bmp";
    const char *outputFile = "output.bmp";
    const char *refFile = "ref.bmp";
    char *filterFile = "filter3.csv";

    // read filter data
    int filterWidth;
    float *filter = readFilter(filterFile, &filterWidth);

    // Homegrown function to read a BMP from file
    float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
    // Size of the input and output images on the host
    int dataSize = imageHeight * imageWidth * sizeof(float);
    // Output image on the host
    float *outputImage = (float *)malloc(dataSize);

    double minThread = 0;
    double recordThread[10] = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(outputImage, 0, dataSize);
        start_time = currentSeconds();
        // Run the host to execute the kernel
        hostFE(inputImage, filter, outputImage, imageHeight, imageWidth, filterWidth);
        end_time = currentSeconds();
        recordThread[i] = end_time - start_time;
    }
    qsort(recordThread, 10, sizeof(double), compare);
    for (int i = 3; i < 7; ++i)
    {
        minThread += recordThread[i];
    }
    minThread /= 4;

    printf("\n[conv cuda]:\t\t[%.3f] ms\n\n", minThread * 1000);

    // Write the output image to file
    storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);

    // Output image of reference on the host
    float *refImage = NULL;
    refImage = (float *)malloc(dataSize);
    memset(refImage, 0, dataSize);

    double minSerial = 0;
    double recordSerial[10] = {0};
    for (int i = 0; i < 10; ++i)
    {
        memset(refImage, 0, dataSize);
        start_time = currentSeconds();
        serialConv(filterWidth, filter, imageHeight, imageWidth, inputImage, refImage);
        end_time = currentSeconds();
        recordSerial[i] = end_time - start_time;
    }
    qsort(recordSerial, 10, sizeof(double), compare);
    for (int i = 3; i < 7; ++i)
    {
        minSerial += recordSerial[i];
    }
    minSerial /= 4;

    printf("\n[conv serial]:\t\t[%.3f] ms\n\n", minSerial * 1000);

    storeImage(refImage, refFile, imageHeight, imageWidth, inputFile);

    int diff_counter = 0;
    for (i = 0; i < imageHeight; i++)
    {
        for (j = 0; j < imageWidth; j++)
        {
            if (abs(outputImage[i * imageWidth + j] - refImage[i * imageWidth + j]) > 10)
            {
                diff_counter += 1;
            }
        }
    }

    float diff_ratio = (float)diff_counter / (imageHeight * imageWidth);
    printf("Diff ratio: %f\n", diff_ratio);

    if (diff_ratio > 0.1)
    {
        printf("\n\033[31mFAILED:\tResults are incorrect!\033[0m\n");
        return -1;
    }
    else
    {
        printf("\n\033[32mPASS:\t(%.2fx speedup over the serial version)\033[0m\n", minSerial / minThread);
    }

    return 0;
}