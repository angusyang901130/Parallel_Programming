#!/bin/sh

# 1.2.1
echo "===== Task 1.2.1 ====="
mpicxx pi_block_linear.cc -o pi_block_linear
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_block_linear ~/pi_block_linear >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_block_linear 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_block_linear 1000000000 2>/dev/null

#1.2.2
echo "===== Task 1.2.2 ====="
mpicxx pi_block_tree.cc -o pi_block_tree
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_block_tree ~/pi_block_tree >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_block_tree 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_block_tree 1000000000 2>/dev/null

#1.2.3
echo "===== Task 1.2.3 ====="
mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_nonblock_linear ~/pi_nonblock_linear >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_nonblock_linear 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_nonblock_linear 1000000000 2>/dev/null

#1.2.4
echo "===== Task 1.2.4 ====="
mpicxx pi_gather.cc -o pi_gather
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_gather ~/pi_gather >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_gather 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_gather 1000000000 2>/dev/null

#1.2.5
echo "===== Task 1.2.5 ====="
mpicxx pi_reduce.cc -o pi_reduce
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_reduce ~/pi_reduce >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_reduce 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_reduce 1000000000 2>/dev/null

#1.2.6
echo "===== Task 1.2.6 ====="
mpicxx pi_one_side.cc -o pi_one_side
parallel-scp -h hosts -r ~/Parallel_Programming/HW4/src/pi_one_side ~/pi_one_side >/dev/null
echo "<<stu>>"
mpirun -np 4 --hostfile hosts pi_one_side 1000000000 2>/dev/null
echo "<<ref>>"
mpirun -np 4 --hostfile hosts /home/HW4/ref/pi_one_side 1000000000 2>/dev/null