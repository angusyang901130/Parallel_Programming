__kernel void convolution(__global float* inputImage, 
                          __global float* filter, 
                          __global float* outputImage, 
                          int imageHeight,
                          int imageWidth, 
                          int filterWidth) 
{   
    int imageX = get_global_id(0);
    int imageY = get_global_id(1);
    
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
