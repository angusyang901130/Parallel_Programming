CC = mpicxx

all: matmul

main:
	$(CC) main.cc -o main

matmul_funct:
	$(CC) matmul_funct.cc -o matmul_funct

matmul: main matmul_funct
	$(CC) main matmul_funct -o matmul

clean:
	rm -f main matmul_funct matmul
