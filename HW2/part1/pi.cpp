#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <climits>
#include <iomanip>
#include <cmath>

using namespace std;

int n_thread;
long long n_toss;
long long n_in_circle = 0;
long long n_toss_per_thread;

pthread_mutex_t mutex;

void* random_toss(void* rank) {
    long long cur_rank = (long long)rank;

    long long local_n_in_circle = 0;
    unsigned int seed = time(NULL);

    long bias = ceil(RAND_MAX / 2.0);

    long value_x, value_y;
    double v_x, v_y;
    double dist_square;

    long long iter_start = cur_rank * n_toss_per_thread;
    long long iter_fin = iter_start + n_toss_per_thread <= n_toss ? iter_start + n_toss_per_thread : n_toss;

    for (long long toss = iter_start; toss < iter_fin; toss++) {

        value_x = rand_r(&seed);
        value_y = rand_r(&seed);

        v_x = (value_x - bias) / (double)bias;
        v_y = (value_y - bias) / (double)bias;

        dist_square = v_x * v_x + v_y * v_y;

        if(dist_square <= 1)
            local_n_in_circle++;
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

    // cout << "Estimated Value of PI: " << setprecision(8) << pi_estimate << endl;
    cout << setprecision(8) << pi_estimate << endl;

    return 0;
}