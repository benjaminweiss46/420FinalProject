
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <chrono>

#define VERTICES 2048
#define SOURCE 6

int main()
{
    int* adj_matrix = (int*)malloc(VERTICES * VERTICES * sizeof(int));  // graph

    // INITIALIZATION

    /*for (int i = 0; i < VERTICES; i++) {
        printf("| %d |", i);
    }
    printf("\n");*/

    for (int a = 0; a < VERTICES; a++)
    {
        //printf("%d: [", a + 1);
        for (int b = 0; b < VERTICES; b++)
        {
            if ((rand() % 2) == 1) { //add connection with random weight between nodes a and b
                adj_matrix[a * VERTICES + b] = rand() % 10;
            }
            else {
                adj_matrix[a * VERTICES + b] = -1;
            }
            //printf("%d, ", adj_matrix[a * VERTICES + b]);
        }
        //printf("]\n");
    }

    int source = SOURCE;     // select intial source vertex
    int dist[VERTICES];      // holds the distance from source vertext to all others
    int included[VERTICES];  // vertexes that are locked in or not

    for (int i = 0; i < VERTICES; i++) { //initialize
        dist[i] = INT_MAX;   // weights are set to max
        included[i] = false;
    }

    dist[source] = 0; //distance to itself is 0
    int min = INT_MAX;
    int minInd;

    // START RUNTIME MEASUREMENT
    printf("Timer starts\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < VERTICES - 1; c++) {
        for (int d = 0; d < VERTICES; d++) {
            if (included[d] == false && dist[d] <= min) {
                min = dist[d];
                minInd = d;
            }
        }
        included[minInd] = true;
        for (int e = 0; e < VERTICES; e++) {
            if (!included[e] && adj_matrix[minInd * VERTICES + e] != -1 && dist[minInd] + adj_matrix[minInd * VERTICES + e] < dist[e] && dist[minInd] != INT_MAX) {
                dist[e] = dist[minInd] + adj_matrix[minInd * VERTICES + e];
            }
        }
    }

    // END RUNTIME MEASUREMENT
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Timer ends\n");
    printf("The sequential Dijkstra's SSSP algorithm took %f seconds to execute graph with %d vertices", elapsed / 1000000000.0, VERTICES);

    // rint distance of source to every vertex
    for (int f = 0; f < VERTICES; f++) {
        if (dist[f] == INT_MAX) {
            dist[f] = -1;
        }
        //printf("Distance to %d is %d\n", f, dist[f]);
    }

    // Free memory
    free(adj_matrix);

    return 0;
}