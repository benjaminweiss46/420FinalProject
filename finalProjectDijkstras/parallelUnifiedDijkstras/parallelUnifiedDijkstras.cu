
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <chrono>

#define VERTICES 1000
#define NUM_OF_THREADS 1024
#define BLOCKS_THREAD  32
#define SOURCE 6 

void write_file(int* output, int number_of_lines, const char* filename);

__global__ void parallelDjikstra(int* adj_matrix, int minInd, bool* included, int* dist) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < VERTICES; idx += blockDim.x * gridDim.x) {
        if (!included[idx] && adj_matrix[minInd * VERTICES + idx] && dist[minInd] + adj_matrix[minInd * VERTICES + idx] < dist[idx] && dist[minInd] != INT_MAX) {
            dist[idx] = dist[minInd] + adj_matrix[minInd * VERTICES + idx];
        }
    }
}

int main()
{
    printf("Vertices: %d\nSource: %d\n", VERTICES, SOURCE);

    //DECLARE VARIABLES 
    cudaError_t cudaStatus = cudaSuccess;
    int* adj_matrix = NULL; //graph
    int* dist = NULL;       //holds the distance from source vertex to all others
    bool* included = NULL;  //vertices that are locked in or not

    //ALLOCATE UNIFIED MEMORY
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

    //INITIALIZATION
    for (int a = 0; a < VERTICES; a++)
    {
        //printf("%d: [", a + 1);
        for (int b = 0; b < VERTICES; b++)
        {
            if ((rand() % 2) == 1) { //add connection with random weight between nodes a and b
                adj_matrix[a * VERTICES + b] = rand() % 10;
            }
            else {
                adj_matrix[a * VERTICES + b] = 0;
            }
            /*if (b < VERTICES - 1) {
                printf("%d, ", adj_matrix[a * VERTICES + b]);
            }
            else {
                printf("%d]\n", adj_matrix[a * VERTICES + b]);
            }*/
        }
    }

    for (int i = 0; i < VERTICES; i++) { //initialize distances
        dist[i] = INT_MAX;   //weights are set to infinity
        included[i] = false;
    }

    dist[SOURCE] = 0; //distance of source to itself is 0

    //START RUNTIME MEASUREMENT
    auto start = std::chrono::high_resolution_clock::now();
    printf("Timer starts\n");

    //Find shortest path for all vertices
    for (int c = 0; c < VERTICES - 1; c++) {
        int min = INT_MAX;
        int minInd;

        //Find vertex with minimum distance from vertices that aren't included
        for (int d = 0; d < VERTICES; d++) {
            if (included[d] == false && dist[d] <= min) {
                min = dist[d];
                minInd = d;
            }
        }
        included[minInd] = true; //vertex is now locked in
        parallelDjikstra << < BLOCKS_THREAD, NUM_OF_THREADS >> > (adj_matrix, minInd, included, dist); //update adjacent vertices

        //Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelDjikstra launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        //cudaDeviceSynchronize waits for the kernel to finish, and returns
        //any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelDjikstra!\n", cudaStatus);
            goto Error;
        }
    }

    //END RUNTIME MEASUREMENT
    printf("Timer ends\n");
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("The sequential Dijkstra's SSSP algorithm took %f ms to execute graph with %d vertices\n", elapsed / 1000000.0, VERTICES);

    //Print distance of source to every vertex
    /*printf("Source -> Vertex: Distance\n");
    for (int f = 0; f < VERTICES; f++) {
        printf("%d -> %d: \t%d\n", SOURCE, f, dist[f]);
    }*/

    const char* output_filename = "parallel_unified_output.txt";
    printf("Writing distance output to %s\n", output_filename);
    write_file(dist, VERTICES, output_filename);

    //cudaDeviceReset must be called before exiting in order for profiling and
    //tracing tools such as Nsight and Visual Profiler to show complete traces.
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

/*
 * Write output file in specified format.
 */
void write_file(int* output, int number_of_lines, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Couldn't open file for writing\n");
        exit(1);
    }

    for (int i = 0; i < number_of_lines; i++) {
        fprintf(fp, "%d\n", output[i]);
    }

    fclose(fp);
}