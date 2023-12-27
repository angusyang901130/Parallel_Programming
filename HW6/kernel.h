#ifndef KERNEL_H_
#define KERNEL_H_

//extern "C"
void cudahostFE(float* inputImage, float* filter, float* outputImage, int imageHeight, int imageWidth, int filterWidth);

#endif /* KERNEL_H_ */