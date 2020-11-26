
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <chrono>

#define VERTICES 2048
#define NUM_OF_THREADS 1024
#define BLOCKS_THREAD  32
#define SOURCE 6 

__global__ void parallelDjikstra(int* adj_matrix, int minInd, bool* included, int* dist) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < VERTICES; idx += blockDim.x * gridDim.x) {
        if (!included[idx] && adj_matrix[minInd * VERTICES + idx] != -1 && dist[minInd] + adj_matrix[minInd * VERTICES + idx] < dist[idx] && dist[minInd] != INT_MAX) {
            dist[idx] = dist[minInd] + adj_matrix[minInd * VERTICES + idx];
        }
    }
}

int main()
{
    // DECLARE VARIABLES 
    cudaError_t cudaStatus = cudaSuccess;
    int* adj_matrix = NULL; // graph
    int* dist = NULL;       // holds the distance from source vertext to all others
    bool* included = NULL;  // vertexes that are locked in or not

    // ALLOCATE UNIFIED MEMORY
    cudaStatus = cudaMallocManaged(&adj_matrix, VERTICES * VERTICES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged1 failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMallocManaged(&dist, VERTICES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged2 failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMallocManaged(&included, VERTICES * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged3 failed!: %s\n", cudaGetErrorString(cudaStatus));
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
        parallelDjikstra << < BLOCKS_THREAD, NUM_OF_THREADS >> > (adj_matrix, minInd, included, dist);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelDjikstra launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelDjikstra!\n", cudaStatus);
            goto Error;
        }
    }

    // END RUNTIME MEASUREMENT
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Timer ends\n");
    printf("The parallel Dijkstra's SSSP algorithm took %f seconds to execute graph with %d vertices using %d threads and %d blocks", elapsed / 1000000000.0, VERTICES, NUM_OF_THREADS, BLOCKS_THREAD);
    
    //Print distance of source to every vertex
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
    cudaFree(adj_matrix);
    cudaFree(dist);
    cudaFree(included);

    return cudaStatus;
}
