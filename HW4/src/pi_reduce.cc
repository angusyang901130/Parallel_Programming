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
    int dst = 0;
    int tag = 0;

    long int count = 0;
    long int reduced_count = 0;
    double rand_bias = RAND_MAX / 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned int seed = world_rank;

    double x, y, dist_sq;

    // TODO: use MPI_Reduce
    for(int i = world_rank; i < tosses; i += world_size){
        x = (rand_r(&seed) - rand_bias) / rand_bias;
        y = (rand_r(&seed) - rand_bias) / rand_bias;

        dist_sq = x * x + y * y;

        if(dist_sq <= 1)
            count++;
    }
    

    MPI_Reduce(&count, &reduced_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * reduced_count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
