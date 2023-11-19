#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

	int numNodes = num_nodes(g);
	// printf("numNodes: %d\n", numNodes);

	double equal_prob = 1.0 / numNodes;

	#pragma omp parallel for
	for (int i = 0; i < numNodes; ++i){
		solution[i] = equal_prob;
	}

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  
	int converged = 0;
	double* new_solution = (double*)malloc(sizeof(double) * g->num_nodes);
	double global_diff;

	while(!converged){

		global_diff = 0.0;

		#pragma omp parallel for
		for(int v = 0; v < numNodes; v++){
			new_solution[v] = 0.0;
		}
		
		#pragma omp parallel for
		for(int v = 0; v < numNodes; v++)
		{

			const Vertex* v_in = incoming_begin(g, v);
			int num_v_in = incoming_size(g, v);

			// double tmp_sol = 0.0;

			// #pragma omp parallel for reduction (+:tmp_sol)
			for(int i = 0; i < num_v_in; i++){
				new_solution[v] += solution[v_in[i]] / outgoing_size(g, v_in[i]);
			}

			new_solution[v] = damping * new_solution[v] + (1.0 - damping) / numNodes;
		}

		double no_outlink_sol = 0.0;

		#pragma omp parallel for reduction (+:no_outlink_sol)
		for(int v = 0; v < numNodes; v++){
			no_outlink_sol += (!outgoing_size(g, v)) * damping * solution[v] / numNodes;
		}

		#pragma omp parallel for 
		for(int v = 0; v < numNodes; v++){
			new_solution[v] += no_outlink_sol;
		}
		
		#pragma omp parallel for reduction (+:global_diff)
		for(int v = 0; v < numNodes; v++){
			global_diff += abs(new_solution[v] - solution[v]);
		}

		// printf("global_diff: %.10lf\n", global_diff);

		converged = (global_diff < convergence);

		#pragma omp parallel for
		for(int v = 0; v < numNodes; v++){
			solution[v] = new_solution[v];
		}

		// double* tmp_ptr = solution;
		// solution = new_solution;
		// new_solution = tmp_ptr;
	}

	free(new_solution);

}
