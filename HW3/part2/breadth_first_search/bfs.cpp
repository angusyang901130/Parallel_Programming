#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define DECISION_VALUE 4

// #define VERBOSE

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{

    vertex_set* local_frontier[4];

    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        vertex_set list;
        vertex_set_init(&list, g->num_nodes);

        local_frontier[id] = &list;

        int count = 0;

        #pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {
            // printf("id: %d, i: %d\n", id, i);
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
                int outgoing = g->outgoing_edges[neighbor];
                
                if (distances[outgoing] == NOT_VISITED_MARKER){
                    
                    distances[outgoing] = distances[node] + 1;
                    int index = count++;
                    local_frontier[id]->vertices[index] = outgoing;

                }
            }
        }

        local_frontier[id]->count = count;

    }

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < local_frontier[i]->count; j++){
            new_frontier->vertices[new_frontier->count++] = local_frontier[i]->vertices[j];
        }
    }
    
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void down_top_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{

    int* in_frontier = (int*)malloc(sizeof(int) * g->num_nodes);

    // vertex_set* in_frontier = &list;

    #pragma omp parallel for
    for(int i = 0; i < g->num_nodes; i++)
        in_frontier[i] = 0;

    #pragma omp parallel for
    for(int i = 0; i < frontier->count; i++)
        in_frontier[frontier->vertices[i]] = 1;

    vertex_set* local_frontier[4];

    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        vertex_set list;
        vertex_set_init(&list, g->num_nodes);

        local_frontier[id] = &list;

        int count = 0;

        #pragma omp for schedule(auto)
        for (int v = 0; v < g->num_nodes; v++){

            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[v + 1];

            if(distances[v] != NOT_VISITED_MARKER)
                continue;
            
            // attempt to add all neighbors to the new frontier
            
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
                int incoming = g->incoming_edges[neighbor];
                
                if (in_frontier[incoming]){
                    // __sync_bool_compare_and_swap(&distances[v], NOT_VISITED_MARKER, distances[outgoing]+1);

                    distances[v] = distances[incoming] + 1;
                    int index = count++;
                    local_frontier[id]->vertices[index] = v;

                    break;
                }
            }
        }

        local_frontier[id]->count = count;

    }
    

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < local_frontier[i]->count; j++){
            new_frontier->vertices[new_frontier->count++] = local_frontier[i]->vertices[j];
        }
    }

    free(in_frontier);
    
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        down_top_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

}

void hybrid_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{   

    if(frontier->count < (double)g->num_nodes / DECISION_VALUE){
        top_down_step(g, frontier, new_frontier, distances);
    }else{
        down_top_step(g, frontier, new_frontier, distances);
    }
    

}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int cur_state = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        hybrid_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

}
