#include <random>
#include <stdio.h>
#include <chrono>

#define VERTICES 1000
#define SOURCE 6

void write_file(int* output, int number_of_lines, const char* filename);

int main()
{
    printf("Vertices: %d\nSource: %d\n", VERTICES, SOURCE);
    int* adj_matrix = (int*)malloc(VERTICES * VERTICES * sizeof(int));  // graph

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

    int dist[VERTICES];      //holds the distance from source vertex to all others
    int included[VERTICES];  //vertices that are locked in or not

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
        for (int e = 0; e < VERTICES; e++) { //update adjacent vertices
            if (!included[e] && adj_matrix[minInd * VERTICES + e] && dist[minInd] + adj_matrix[minInd * VERTICES + e] < dist[e] && dist[minInd] != INT_MAX) {
                dist[e] = dist[minInd] + adj_matrix[minInd * VERTICES + e];
            }
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

    const char* output_filename = "sequential_output.txt";
    printf("Writing distance output to %s\n", output_filename);
    write_file(dist, VERTICES, output_filename);

    //Free memory
    free(adj_matrix);

    return 0;
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