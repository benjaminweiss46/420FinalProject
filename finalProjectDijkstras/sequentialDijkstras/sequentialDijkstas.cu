
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <time.h>
const int n = 50000;



int main()
{
    //int adj_matrix[n][n];
    int* adj_matrix[n];
    for (int i = 0; i < n; i++) {
        adj_matrix[i] = (int*)malloc(n * sizeof(int));
    }
    for (int i = 0; i < n; i++) {
        //printf("| %d |", i);
    }
    //printf("\n");
    for (int a = 0; a < n; a++)
    {
        //printf("%d: [", a + 1);
        for (int b = 0; b < n; b++)
        {
            if ((rand() % 2) == 1) { //add connection with random weight between nodes a and b
                adj_matrix[a][b] = rand() % 10;
            }
            else {
                adj_matrix[a][b] = -1;
            }
            //printf("%d, ", adj_matrix[a][b]);
        }
        //printf("]\n");
    }
    clock_t t;
    t = clock();
    printf("Timer starts\n");
    int source = 6; //change to find the source for other points
    int dist[n]; //holds the distance from source vertext to all others
    bool included[n]; //vertexes that are locked in or not
    for (int i = 0; i < n; i++) { //initialize
        dist[i] = INT_MAX;
        included[i] = false;
    }
    dist[source] = 0; //distance to itself is 0
    int min = INT_MAX;
    int minInd;
    for (int c = 0; c < n - 1; c++) {
        for (int d = 0; d < n; d++) {
            if (included[d] == false && dist[d] <= min) {
                min = dist[d];
                minInd = d;
            }
        }
        included[minInd] = true;
        for (int e = 0; e < n; e++) {
            if (!included[e] && adj_matrix[minInd][e] != -1 && dist[minInd] + adj_matrix[minInd][e] < dist[e] && dist[minInd] != INT_MAX) {
                dist[e] = dist[minInd] + adj_matrix[minInd][e];
            }
        }
    }
    for (int f = 0; f < n; f++) {
        if (dist[f] == INT_MAX) {
            dist[f] = -1;
        }
        //printf("Distance to %d is %d\n", f, dist[f]);
    }
    printf("Timer ends \n");
    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC; // calculate the elapsed time
    printf("The program took %f seconds to execute", time_taken);

    return 0;
}
