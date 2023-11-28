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

    int dst = 0;
    int tag = 0;

    long int count = 0;
    double rand_bias = RAND_MAX / 2;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned int seed = world_rank;

    double x, y, dist_sq;


    if (world_rank > 0)
    {
        // TODO: MPI workers

        MPI_Request req;

        for(int i = world_rank; i < tosses; i += world_size){
            x = (rand_r(&seed) - rand_bias) / rand_bias;
            y = (rand_r(&seed) - rand_bias) / rand_bias;

            dist_sq = x * x + y * y;

            if(dist_sq <= 1)
                count++;
        }

        
        // printf("rank: %d, count: %ld\n", world_rank, count);
        MPI_Isend(&count, 1, MPI_LONG, dst, tag, MPI_COMM_WORLD, &req);
        // MPI_Wait(&req, &status);
        
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        MPI_Request requests[LEN];
        long int recv_cnt[LEN];

        // MPI_Status statuses[LEN];

        for(int i = 0; i < tosses; i += world_size){
            x = (rand_r(&seed) - rand_bias) / rand_bias;
            y = (rand_r(&seed) - rand_bias) / rand_bias;

            dist_sq = x * x + y * y;

            if(dist_sq <= 1)
                count++;
        }
        
        for(int src = 1; src < world_size; src++){
            MPI_Irecv(&recv_cnt[src], 1, MPI_LONG, src, tag, MPI_COMM_WORLD, &requests[src]);
        }

        MPI_Waitall(world_size-1, requests+1, MPI_STATUS_IGNORE);

        for(int src = 1; src < world_size; src++){
            // printf("src: %d, count: %ld\n", src, recv_cnt[src]);
            count += recv_cnt[src];
        }
        
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
