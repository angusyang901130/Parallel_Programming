make >/dev/null
parallel-scp -h ../host.txt -r ~/Parallel_Programming/HW4/part2/matmul ~/matmul >/dev/null
parallel-scp -h ../host.txt -r ~/Parallel_Programming/HW4/part2/test.txt ~/test.txt >/dev/null
mpirun -np 4 --hostfile hosts matmul < /home/.grade/HW4/data-set/data1_1 2>/dev/null 1>output.txt