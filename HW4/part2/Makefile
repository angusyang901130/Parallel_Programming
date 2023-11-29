CC = mpicxx

all: matmul

matmul: main matmul_funct
	$(CC) main matmul_funct -o matmul

main: main.cc
	$(CC) -c main.cc -o main

matmul_funct: matmul_funct.cc
	$(CC) -c matmul_funct.cc -o matmul_funct

clean:
	rm -f main matmul_funct matmul
