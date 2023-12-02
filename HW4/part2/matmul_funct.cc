#include <cstdio>
#include <mpi.h>

#define LEN 20

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{

    int world_rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Status status;

    int src = 0;
    int tag = 0;

    int n, m, l;

    if(world_rank == 0){

        scanf("%d %d %d\n", n_ptr, m_ptr, l_ptr);

        n = *n_ptr;
        m = *m_ptr; 
        l = *l_ptr;

        *a_mat_ptr = (int*)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int*)malloc(m * l * sizeof(int));

        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                int pos = i * m + j;
                if(j == m-1){
                    scanf("%d\n", (*a_mat_ptr)+pos);
                }else{
                    scanf("%d ", (*a_mat_ptr)+pos);
                }
            }
        }

        for(int i = 0; i < m; i++){
            for(int j = 0; j < l; j++){
                int pos = j * m + i;
                if(j == m-1){
                    scanf("%d\n", (*b_mat_ptr)+pos);
                }else{
                    scanf("%d ", (*b_mat_ptr)+pos);
                }
            }
        }

        for(int dst = 1; dst < world_size; dst++){
            MPI_Send(n_ptr, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
        }

        for(int dst = 1; dst < world_size; dst++){
            MPI_Send(m_ptr, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
        }

        for(int dst = 1; dst < world_size; dst++){
            MPI_Send(l_ptr, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
        }

        // for(int dst = 1; dst < world_size; dst++){
        //     MPI_Send(*a_mat_ptr, n * l, MPI_INT, dst, 0, MPI_COMM_WORLD);
        // }

    }else{
        MPI_Recv(n_ptr, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(m_ptr, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(l_ptr, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);

        n = *n_ptr;
        m = *m_ptr;
        l = *l_ptr;
    }

    // printf("world_rank: %d, n: %d, m:%d, l:%d\n", world_rank, n, m, l);
    
    
    if(world_rank == 0){

        MPI_Request requests[LEN];

        for(int dst = 1; dst < world_size; dst++){
            MPI_Isend(*a_mat_ptr, n*m, MPI_INT, dst, tag, MPI_COMM_WORLD, &requests[dst]);
        }

    }else{
        MPI_Request req;

        *a_mat_ptr = (int*)malloc(n * m * sizeof(int));
        *b_mat_ptr = (int*)malloc(m * l * sizeof(int));
        
        MPI_Irecv(*a_mat_ptr, n*m, MPI_INT, src, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

    }

    
    if(world_rank == 0){

        MPI_Request requests[LEN];

        for(int dst = 1; dst < world_size; dst++){
            MPI_Isend(*b_mat_ptr, m*l, MPI_INT, dst, tag, MPI_COMM_WORLD, &requests[dst]);
        }

    }else{
        MPI_Request req;

        MPI_Irecv(*b_mat_ptr, m*l, MPI_INT, src, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        
    }

    // if(world_rank != 0){
    //     for(int i = 0; i < n; i++){
    //         for(int j = 0; j < m; j++){
    //             int pos = i * m + j;
    //             printf("%d ", (*a_mat_ptr)[pos]);
    //         }
    //         printf("\n");
    //     }
        

    //     for(int i = 0; i < l; i++){
    //         for(int j = 0; j < m; j++){
    //             int pos = i * m + j;
    //             printf("%d ", (*b_mat_ptr)[pos]);
    //         }
    //         printf("\n");
    //     }
    // }
    

}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;

    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int* c_mat = (int*)calloc(n * l, sizeof(int));
    int* res_c_mat = (int*)calloc(n * l, sizeof(int));

    // printf("n: %d, m: %d, l: %d\n", n, m, l);

    int col_per_rank = 0;

    if(m % world_size == 0){
        // printf("world_rank: %d, m: %d, world_size: %d\n", world_rank, m, world_size);
        col_per_rank = m / world_size;
    }else{
        col_per_rank = m / world_size + 1;
    }

    // printf("world_rank: %d, col_per_rank: %d\n", world_rank, col_per_rank);
    int col_start = world_rank * col_per_rank; 
    // printf("col_per_rank * world_rank + col_per_rank: %d\n", col_per_rank * (world_rank + 1));
    int col_end = (world_rank + 1) * col_per_rank;

    if(m < col_end)
        col_end = m;

    // printf("world_rank: %d, col_start: %d, col_end: %d\n", world_rank, col_start, col_end);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < l; j++){
            for(int k = col_start; k < col_end; k++){
                // printf("world_rank: %d, a: %d, b: %d\n", world_rank, a_mat[i*m+k], b_mat[j*m+k]);
                c_mat[i*l + j] += a_mat[i*m+k] * b_mat[j*m+k];
            }
        }
    }

    MPI_Reduce(c_mat, res_c_mat, n*l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < l; j++){
                if(j == l-1)
                    printf("%d\n", res_c_mat[i*l + j]);
                else printf("%d ", res_c_mat[i*l + j]);
            }
        }
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
}