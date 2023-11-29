#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Status status;
    unsigned int seed = world_rank;
    int tag = 0;

    long int count = 0;
    double rand_bias = RAND_MAX / 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // TODO: binary tree redunction
    int tmp_rank = world_rank;
    int level = 1;

    double x, y, dist_sq;
    for(int i = world_rank; i < tosses; i += world_size){
        x = (rand_r(&seed) - rand_bias) / rand_bias;
        y = (rand_r(&seed) - rand_bias) / rand_bias;

        dist_sq = x * x + y * y;

        if(dist_sq <= 1)
            count++;
    }

    while(tmp_rank % 2 == 0 && level < world_size){
        // printf("world rank: %d, level: %d, recv_rank: %d, tmp_rank: %d\n", world_rank, level, world_rank+level, tmp_rank);

        long int recv_cnt;
        MPI_Recv(&recv_cnt, 1, MPI_LONG, world_rank+level, tag, MPI_COMM_WORLD, &status);

        count += recv_cnt;

        tmp_rank = tmp_rank >> 1;
        level = level << 1;
    }

    if(world_rank > 0){
        // printf("world_rank: %d\n", world_rank);
        MPI_Send(&count, 1, MPI_LONG, world_rank-level, tag, MPI_COMM_WORLD);
    }
    
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
