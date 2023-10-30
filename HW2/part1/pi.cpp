#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <cmath>
#include <climits>
#include <iomanip>

using namespace std;


int n_thread;
long long n_toss;
long long n_in_circle = 0;
long long n_toss_per_thread;

pthread_mutex_t mutex;

void* random_toss(void* rank){
    long long cur_rank = (long long)rank;
    // cout << "cur_rank: " << cur_rank << endl;

    long long local_n_in_circle = 0;
    double dist_squared = 0;

    struct drand48_data buffer;
    srand48_r(time(NULL), &buffer);

    long value_x;
    long value_y;

    long long iter_start = cur_rank * n_toss_per_thread;
    long long iter_fin = iter_start + n_toss_per_thread <= n_toss ? iter_start + n_toss_per_thread : n_toss;

    for(long long toss = iter_start; toss < iter_fin; toss++){
        mrand48_r(&buffer, &value_x);
        mrand48_r(&buffer, &value_y);

        // if(value_x > INT_MAX || value_x < -INT_MAX || value_y > INT_MAX || value_y < -INT_MAX)
        //     cout << "range error !!" << endl;

        double x = ((double)value_x) / INT_MAX;
        double y = ((double)value_y) / INT_MAX;
        
        dist_squared = x * x + y * y;

        if (dist_squared <= 1)
            local_n_in_circle++;
    }

    pthread_mutex_lock(&mutex);
    n_in_circle += local_n_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;

}

int main(int argc, char* argv[]){

    if(argc != 3){
        cout << "Argument is not enough, exiting ..." << endl;
        return 0;
    }

    long thread;

    n_toss = strtoll(argv[2], NULL, 10);
    // cout << "n_toss: " << n_toss << endl;

    n_in_circle = 0;
    n_thread = atoi(argv[1]);
    // cout << "n_thread: " << n_thread << endl;

    n_toss_per_thread = ceil(n_toss / (double)n_thread);
    // cout << "n_toss_per_thread: " << n_toss_per_thread << endl;

    pthread_t* thread_handles = (pthread_t*)calloc(n_thread, sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    for(thread = 0; thread < n_thread; thread++)
        pthread_create(&thread_handles[thread], NULL, random_toss, (void*)thread);

    for(thread = 0; thread < n_thread; thread++)
        pthread_join(thread_handles[thread], NULL);

    double pi_estimate = 4 * n_in_circle /(( double ) n_toss);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    cout << setprecision(8) << pi_estimate << endl;

    return 0;
}


