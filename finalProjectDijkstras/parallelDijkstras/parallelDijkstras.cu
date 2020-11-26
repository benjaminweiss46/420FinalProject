
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <chrono>

#define VERTICES 2048
#define NUM_OF_THREADS 1024
#define BLOCKS_THREAD  32
#define SOURCE 6 

__global__ void parallelDjikstra(int* minInd, bool* included, int* dist, int* adj_matrix) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < VERTICES; idx += blockDim.x * gridDim.x) {
        if (!included[idx] && adj_matrix[*minInd * VERTICES + idx] != -1 && dist[*minInd] 
            + adj_matrix[*minInd * VERTICES + idx] < dist[idx] && dist[*minInd] != INT_MAX) {
            dist[idx] = dist[*minInd] + adj_matrix[*minInd * VERTICES + idx];
        }
    }
}

__global__ void minDist(int* minInd, bool* included, int* dist, int* min) {
    for (int d = 0; d < VERTICES; d++) {
        if (included[d] == false && dist[d] <= *min) {
            *min = dist[d];
            *minInd = d;
        }
    }
    included[*minInd] = true;
}

int main()
{
    // DECLARE VARIABLES 
    cudaError_t cudaStatus = cudaSuccess;
    int* adj_matrix = (int*)malloc(VERTICES * VERTICES * sizeof(int)); // graph
    int* dist = (int*)malloc(VERTICES * sizeof(int));                  // holds the distance from source vertext to all others
    bool* included = (bool*)malloc(VERTICES * sizeof(bool));           // vertexes that are locked in or not
    int* min = (int*)malloc(sizeof(int));
    int* minInd = (int*)malloc(sizeof(int));

    // device variables
    int* d_adj_matrix = NULL;
    int* d_dist = NULL;
    bool* d_included = NULL;
    int* d_min = NULL;
    int* d_minInd = NULL;

    // ALLOCATE EXPLICIT MEMORY
    cudaStatus = cudaMalloc(&d_adj_matrix, VERTICES * VERTICES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_dist, VERTICES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_included, VERTICES * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_min, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_minInd, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

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

    int source = SOURCE; //select intial source vertex

    for (int i = 0; i < VERTICES; i++) { //initialize
        dist[i] = INT_MAX;   // weights are set to max
        included[i] = false;
    }

    dist[source] = 0; //distance to itself is 0
    *min = INT_MAX;

    cudaStatus = cudaMemcpy(d_adj_matrix, adj_matrix, VERTICES * VERTICES * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_included, included, VERTICES * sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_dist, dist, VERTICES * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_min, min, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // START RUNTIME MEASUREMENT
    printf("Timer starts\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < VERTICES - 1; c++) {
        minDist << < 1, 1 >> > (d_minInd, d_included, d_dist, d_min);
        parallelDjikstra << < BLOCKS_THREAD, NUM_OF_THREADS >> > (d_minInd, d_included, d_dist, d_adj_matrix);
    }

    // END RUNTIME MEASUREMENT
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Timer ends\n");
    printf("The parallel Dijkstra's SSSP algorithm took %f seconds to execute graph with %d vertices using %d threads and %d blocks\n", elapsed / 1000000000.0, VERTICES, NUM_OF_THREADS, BLOCKS_THREAD);

    // Copy dist back to host to print
    cudaStatus = cudaMemcpy(dist, d_dist, VERTICES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    /*
    // Test
    // copy graph back to host for testing
    cudaStatus = cudaMemcpy(adj_matrix, d_adj_matrix, VERTICES * VERTICES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    for (int i = 0; i < VERTICES; i++)
        for (int j = 0; j < VERTICES; j++)
            printf("adj_matrix[%d][%d]: %d\n", i, j, adj_matrix[i * VERTICES + j]);

    for (int i = 0; i < VERTICES; i++)
        printf("dist[%d]: %d\n", i, dist[i]);*/

    // Print distance of source to every vertex
    for (int f = 0; f < VERTICES; f++) {
        if (dist[f] == INT_MAX) {
            dist[f] = -1;
        }
        //printf("Distance to %d is %d\n", f, dist[f]);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

Error:
    //Cleanup
    cudaFree(d_adj_matrix);
    cudaFree(d_dist);
    cudaFree(d_included);
    cudaFree(d_min);
    cudaFree(d_minInd);
    free(adj_matrix);
    free(dist);
    free(included);
    free(min);
    free(minInd);

    return cudaStatus;
}
