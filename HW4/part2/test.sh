make >/dev/null
parallel-scp -h ~/Parallel_Programming/HW4/hosts -r ~/Parallel_Programming/HW4/part2/matmul ~/matmul >/dev/null
parallel-scp -h ~/Parallel_Programming/HW4/hosts -r ~/Parallel_Programming/HW4/part2/test.txt ~/test.txt >/dev/null
mpirun -np 1 --hostfile ~/Parallel_Programming/HW4/hosts matmul < test.txt #2>/dev/null