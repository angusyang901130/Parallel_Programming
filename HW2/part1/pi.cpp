#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

double n_in_circle;
int n_thread;
long long n_toss;
long long n_toss_per_thread;

pthread_mutex_t mutex;

void* random_toss(void* rank){
    long long cur_rank = (long long)rank;
    long long local_n_in_circle = 0;
    double dist_squared = 0;

    struct drand48_data buffer;
    srand48_r(time(NULL), &buffer);
    long int value_x;
    long int value_y;

    for(int toss = rank*n_toss_per_thread; toss < n_toss; toss++){
        x = drand48_r(&buffer, &value_x);
        y = drand48_r(&buffer, &value_y);
        
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

    n_toss = strtoll([argv[2]]);
    n_in_circle = 0;
    n_thread = atoi(argv[1]);
    n_toss_per_thread = ceil(n_toss / (double)n_thread);

    pthread_t* thread_handles = (pthread_t*)calloc(n_thread, sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    for(thread = 0; thread < n_thread; thread++)
        pthread_create(&thread_handles[thread], NULL, random_toss, (void*)thread);

    for(thread = 0; thread < n_thread; thread++)
        pthread_join(thread_handles[thread], NULL);

    pi_estimate = 4 * n_in_circle /(( double ) n_of_toss);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    cout << pi_estimate << endl;

    return 0;
}


