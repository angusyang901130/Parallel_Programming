#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#define LEN 20

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
    int dst = 0;
    int tag = 0;

    long int count = 0;
    double rand_bias = RAND_MAX / 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned int seed = world_rank;

    double x, y, dist_sq;

    long int recv_cnt[LEN];

    // TODO: use MPI_Gather
    for(int i = world_rank; i < tosses; i += world_size){
        x = (rand_r(&seed) - rand_bias) / rand_bias;
        y = (rand_r(&seed) - rand_bias) / rand_bias;

        dist_sq = x * x + y * y;

        if(dist_sq <= 1)
            count++;
    }
    
    MPI_Gather(&count, 1, MPI_LONG, recv_cnt, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // printf("world_rank: %d, count: %d\n", world_rank, count);

    for(int i = 1; i < world_size; i++){
        // printf("recv_cnt[%d] = %ld\n", i, recv_cnt[i]);
        count += recv_cnt[i];
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
