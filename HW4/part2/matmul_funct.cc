#include <cstdio>
#include <mpi.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    scanf("%d %d %d\n", n_ptr, m_ptr, l_ptr);

    int n = *n_ptr;
    int m = *m_ptr; 
    int l = *l_ptr;

    a_mat_ptr = (int**)malloc(n * sizeof(int*));
    b_mat_ptr = (int**)malloc(m * sizeof(int*));

    int* a_cont_space = (int*)malloc(n * m * sizeof(int));
    int* b_cont_space = (int*)malloc(m * l * sizeof(int));

    for(int i = 0; i < n; i++)
        a_mat_ptr[i] = &a_cont_space[i * m];
    
    for(int i = 0; i < m; i++)
        b_mat_ptr[i] = &b_cont_space[i * l];
    
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(j == m-1){
                scanf("%d\n", &a_mat_ptr[i][j]);
            }else{
                scanf("%d ", &a_mat_ptr[i][j]);
            }
        }
    }
    

    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < m; j++){
    //         printf("%d ", a_mat_ptr[i][j]);
    //     }
    //     printf("\n");
    // }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < l; j++){
            int pos = i * l + j;
            if(j == l-1){
                scanf("%d\n", &b_mat_ptr[i][j]);
            }else{
                scanf("%d ", &b_mat_ptr[i][j]);
            }
        }
    }

    // for(int i = 0; i < m; i++){
    //     for(int j = 0; j < l; j++){
    //         printf("%d ", b_mat_ptr[i][j]);
    //     }
    //     printf("\n");
    // }


}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;

    MPI_Status status;
    int dst = 0;
    int tag = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_rank > 0){

    }else if(world_rank == 0){
        
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
}