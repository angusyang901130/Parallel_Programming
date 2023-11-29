#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

long int cnt_funct(int rank, long int size, long int tosses){

    long int count = 0;
    double rand_bias = RAND_MAX / 2;
    double x, y, dist_sq;

    unsigned int seed = rank;

    for(int i = rank; i < tosses; i += size){
        x = (rand_r(&seed) - rand_bias) / rand_bias;
        y = (rand_r(&seed) - rand_bias) / rand_bias;

        dist_sq = x * x + y * y;

        if(dist_sq <= 1)
            count++;
    }

    return count;
}

int is_ready(long int* recv_cnt, int size){

    int cnt = 0;
    for(int i = 1; i < size; i++){
        if(recv_cnt[i] != -1)
            cnt++;
    }

    return (cnt == size-1);
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Status status;
    int dst = 0;
    int tag = 0;
    long int count = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    if (world_rank == 0)
    {
        // Master
        long int* recv_cnt;
        MPI_Alloc_mem(world_size * sizeof(long int), MPI_INFO_NULL, &recv_cnt);

        for (int i = 0; i < world_size; i++){
            recv_cnt[i] = -1;
        }

        count = cnt_funct(world_rank, world_size, tosses);
        // printf("rank: %d -> %ld\n", world_rank, count);

        MPI_Win_create(recv_cnt, world_size * sizeof(long int), sizeof(long int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready){
            // Without the lock/unlock schedule stays forever filled with 0s
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = is_ready(recv_cnt, world_size);
            MPI_Win_unlock(0, win);
        }

        for(int i = 1; i < world_size; i++){
            // printf("i: %d -> %ld\n", i, recv_cnt[i]);
            count += recv_cnt[i];
        }

        MPI_Free_mem(recv_cnt);

    }
    else
    {
        // Workers
        count = cnt_funct(world_rank, world_size, tosses);
        // printf("rank: %d -> %ld\n", world_rank, count);

        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&count, 1, MPI_LONG, 0, world_rank, 1, MPI_LONG, win);
        MPI_Win_unlock(0, win);

    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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