default: conv

CC = gcc-10
NVCC = nvcc
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g
NVFLAGS = -O2 -Xcompiler -lrt -lm

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o
CUDA_OBJS = conv_cuda.o bmpfuncs.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

conv_cuda: kernel.cu
	$(NVCC) $(NVFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@


clean:
	rm -f conv *.o output.bmp ref.bmp