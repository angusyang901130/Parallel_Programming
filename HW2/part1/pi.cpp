#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <climits>
#include <iomanip>
#include <cmath>
#include <x86intrin.h>

#pragma GCC target("avx2")

using namespace std;

const int VECTOR_SIZE = 8;

int n_thread;
long long n_toss;
long long n_in_circle = 0;
long long n_toss_per_thread;

pthread_mutex_t mutex;

void* random_toss(void* rank) {
    long long cur_rank = (long long)rank;

    long long local_n_in_circle = 0;

    __m256 v_max = _mm256_set1_ps(1);
    __m256 one_v = _mm256_set1_ps(INT_MAX);

    struct drand48_data buffer;
    srand48_r(time(NULL), &buffer);

    long values_x[VECTOR_SIZE];
    long values_y[VECTOR_SIZE];

    long long iter_start = cur_rank * n_toss_per_thread;
    long long iter_fin = iter_start + n_toss_per_thread <= n_toss ? iter_start + n_toss_per_thread : n_toss;

    for (long long toss = iter_start; toss < iter_fin; toss += VECTOR_SIZE) {
        // Generate VECTOR_SIZE random integers in parallel
        
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            if(toss+i < iter_fin){
                mrand48_r(&buffer, &values_x[i]);
                mrand48_r(&buffer, &values_y[i]);
            }else{
                values_x[i] = INT_MAX;
                values_y[i] = INT_MAX;
            }
        }

        __m256i i32_x = _mm256_set_epi32(values_x[7], values_x[6], values_x[5], values_x[4], values_x[3], values_x[2], values_x[1], values_x[0]);
        __m256i i32_y = _mm256_set_epi32(values_y[7], values_y[6], values_y[5], values_y[4], values_y[3], values_y[2], values_y[1], values_y[0]);

        __m256 ps_x = _mm256_cvtepi32_ps(i32_x);
        __m256 ps_y = _mm256_cvtepi32_ps(i32_y);
        
        // __m256 pd_x = _mm256_cvtps_ps(ps_x);
        // __m256 pd_y = _mm256_cvtps_ps(ps_y);

        __m256 v_x = _mm256_div_ps(ps_x, one_v);
        __m256 v_y = _mm256_div_ps(ps_y, one_v);

        // Convert to doubles and calculate the square of the distances in parallel
        __m256 v_x_squared = _mm256_mul_ps(v_x, v_x);
        __m256 v_y_squared = _mm256_mul_ps(v_y, v_y);
        __m256 v_squared = _mm256_add_ps(v_x_squared, v_y_squared);

        // Check if the squared distances are less than or equal to 1 (in parallel)
        __m256 v_cmp = _mm256_cmp_ps(v_max, v_squared, _CMP_GT_OQ);
        local_n_in_circle += _mm_popcnt_u32(_mm256_movemask_ps(v_cmp));
    }

    pthread_mutex_lock(&mutex);
    n_in_circle += local_n_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Argument is not enough, exiting ..." << endl;
        return 0;
    }

    long thread;

    n_toss = strtoll(argv[2], NULL, 10);
    n_in_circle = 0;
    n_thread = atoi(argv[1]);
    n_toss_per_thread = ceil(n_toss / (double)n_thread);

    pthread_t* thread_handles = (pthread_t*)calloc(n_thread, sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    for (thread = 0; thread < n_thread; thread++)
        pthread_create(&thread_handles[thread], NULL, random_toss, (void*)thread);

    for (thread = 0; thread < n_thread; thread++)
        pthread_join(thread_handles[thread], NULL);

    float pi_estimate = 4 * n_in_circle / ((double)n_toss);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    cout << "Estimated Value of PI: " << setprecision(8) << pi_estimate << endl;

    return 0;
}