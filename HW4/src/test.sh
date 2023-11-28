#!/bin/sh

# 1.2.1
mpicxx pi_block_linear.cc -o pi_block_linear
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_block_linear ~/pi_block_linear
mpirun -np 4 --hostfile hosts pi_block_linear 1000000000