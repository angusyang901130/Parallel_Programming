__kernel void convolution(__global float* inputImage, 
                          __global float* filter, 
                          __global float* outputImage, 
                          int imageHeight,
                          int imageWidth, 
                          int filterWidth) 
{   
    int imageX = get_global_id(0);
    int imageY = get_global_id(1);

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
