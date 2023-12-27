__kernel void convolution(__global float* inputImage, 
                          __global float* filter, 
                          __global float* outputImage, 
                          int imageHeight,
                          int imageWidth, 
                          int filterWidth) 
{   
    int x = get_global_id(0);
    int y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int localDim_x = get_local_size(0);
    int localDim_y = get_local_size(1);

    int halffilterSize = filterWidth / 2;

    int imageX = x * localDim_x + local_x;
    int imageY = y * localDim_y + local_y;
    
    float sum = 0.0f;
    for (i = -halffilterSize; i <= halffilterSize; i++)
    {
        for (j = -halffilterSize; j <= halffilterSize; j++)
        {

            if (imageY + i >= 0 && imageY + i < imageHeight &&
                imageX + j >= 0 && imageX + j < imageWidth)
            {
                sum += inputImage[(imageY + i) * imageWidth + imageX + j] *
                        filter[(i + halffilterSize) * filterWidth +
                                j + halffilterSize];
            }

            
        }
    }
    
    outputImage[imageY * imageWidth + imageX] = sum;
    
}
