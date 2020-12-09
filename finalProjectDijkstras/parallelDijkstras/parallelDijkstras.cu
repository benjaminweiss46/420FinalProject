
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

__global__ void parallelDjikstra(int* minInd, bool* included, int* dist, int* adj_matrix) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < VERTICES; idx += blockDim.x * gridDim.x) {
        if (!included[idx] && adj_matrix[*minInd * VERTICES + idx] && dist[*minInd]
            + adj_matrix[*minInd * VERTICES + idx] < dist[idx] && dist[*minInd] != INT_MAX) {
            dist[idx] = dist[*minInd] + adj_matrix[*minInd * VERTICES + idx];
        }
    }
}

__global__ void minDist(int* minInd, bool* included, int* dist) {
    int min = INT_MAX;
    for (int d = 0; d < VERTICES; d++) {
        if (included[d] == false && dist[d] <= min) {
            min = dist[d];
            *minInd = d;
        }
    }
    included[*minInd] = true;
}

int main()
{
    printf("Vertices: %d\nSource: %d\n# of threads: %d\n# of blocks: %d\n", VERTICES, SOURCE, NUM_OF_THREADS, BLOCKS_THREAD);

    //DECLARE VARIABLES 
    cudaError_t cudaStatus = cudaSuccess;
    int* adj_matrix = (int*)malloc(VERTICES * VERTICES * sizeof(int)); //graph
    int* dist = (int*)malloc(VERTICES * sizeof(int));                  //holds the distance from source vertext to all others
    bool* included = (bool*)malloc(VERTICES * sizeof(bool));           //vertices that are locked in or not
    int* minInd = (int*)malloc(sizeof(int));

    //Device variables
    int* d_adj_matrix = NULL;
    int* d_dist = NULL;
    bool* d_included = NULL;
    int* d_minInd = NULL;

    //ALLOCATE EXPLICIT MEMORY
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

    cudaStatus = cudaMalloc(&d_minInd, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
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

    //START RUNTIME MEASUREMENT
    printf("Timer starts\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < VERTICES - 1; c++) {
        minDist << < 1, 1 >> > (d_minInd, d_included, d_dist); //find vertex with minimum distance from vertices that aren't included
        parallelDjikstra << < BLOCKS_THREAD, NUM_OF_THREADS >> > (d_minInd, d_included, d_dist, d_adj_matrix); //update adjacent vertices
    }

    //END RUNTIME MEASUREMENT
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Timer ends\n");
    printf("The parallel explicit Dijkstra's SSSP algorithm took %f ms.\n", elapsed / 1000000.0);

    //Copy dist back to host to print
    cudaStatus = cudaMemcpy(dist, d_dist, VERTICES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //Print distance of source to every vertex
    /*printf("Source -> Vertex: Distance\n");
    for (int f = 0; f < VERTICES; f++) {
        printf("%d -> %d: \t%d\n", SOURCE, f, dist[f]);
    }*/

    const char* output_filename = "parallel_explicit_output.txt";
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
    cudaFree(d_adj_matrix);
    cudaFree(d_dist);
    cudaFree(d_included);
    cudaFree(d_minInd);
    free(adj_matrix);
    free(dist);
    free(included);
    free(minInd);

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