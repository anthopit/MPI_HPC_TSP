#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stddef.h>
#include <math.h>
#include <mpi.h>
#include <limits.h>

//------------------------------- Distance matrix ----------------------------------//

/**
 * Read the fisrt line of the distance file to get the total number of cities
 * @param filename
 * @return Total number of cities
 */
int getTotalCity(char* filename) {
    int total_city;

    FILE *fp = fopen(filename, "r");

    if (fp == NULL)
    {
        printf("Error: could not open file %s", filename);
    }

    // reading line by line, max 256 bytes
    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];

    fgets(buffer, MAX_LENGTH, fp);

    total_city = atoi(buffer);

    return total_city;
}

/**
 * Create matrix in 1D array shape of distance between cities from the distance file
 * @param filename
 * @param total_city number of cities get with getTotalCity()
 * @return 1D array of distance between cities
 */

int* createDistanceMatrix(char* filename, int total_city) {
    FILE *fp = fopen(filename, "r");

    if (fp == NULL)
    {
        printf("Error: could not open file %s", filename);
    }

    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];

    fgets(buffer, MAX_LENGTH, fp);

    int *distance_matrix;

    distance_matrix = malloc((total_city * total_city) * sizeof(int));

    for (int i = 0; i < (total_city*total_city); i++){
        distance_matrix[i] = 0;
    }

    int row_input_number = 1;
    int column_input_number = 0;
    // reading line by line, max 256 bytes
    while (fgets(buffer, MAX_LENGTH, fp))
    {

        char * token = strtok(buffer, " ");

        // walk through elements of line and add them to distance matrix
        while( token != NULL ) {
            distance_matrix[(row_input_number * total_city) + (column_input_number)] = atoi(token);
            token = strtok(NULL, " ");
            column_input_number += 1;
        }


        column_input_number = 0;
        row_input_number += 1;
    }

    // make the matrix symmetric
    for (int i = 0; i < total_city; i++) {
        for (int j = 0; j < total_city; j++) {
            if (i < j) {
                distance_matrix[i * total_city + j] = distance_matrix[j * total_city + i];
            }
        }
    }

    // close the file
    fclose(fp);

    return distance_matrix;
}

//-------------------------------------- Branch and Bounds ----------------------------------//

struct Branch {
    int *path;
    int distance;
};

/**
 * Check is the path is valid, ie. each city is visited only once
 * @param current_path Path to check
 * @param total_city Total number of cities
 * @return true if the path is valid, false otherwise
 */
bool isPathValid(int *current_path, int total_city){
    // create an array of boolean of size total_city
    bool visited_city[total_city];

    // initialize the array to false
    for (int i = 0; i < total_city; i++) {
        visited_city[i] = false;
    }

    // For each city in the path, check if it has been visited
    for (int i = 0; i < total_city; i++) {
        // if the city has been visited, return false
        if (visited_city[current_path[i]-1] && current_path[i] != 0) {
            return false;
        }
        // if the city is not visited, mark the city as visited
        visited_city[current_path[i]-1] = true;
    }

    return true;
}

/**
 * Check if the distance of the current path is better than the best distance
 * Check if there is a new best distance from the other processes to recevie every 100 calls
 * If the path is not complete (is_path_complete == false)
 * Return true if the path is the best path, false otherwise
 * If the path is complete (is_path_complete == true)
 * Return true if the path is the best path, false otherwise
 * Send the distance to the all processes and update the best distance
 * @param distance_matrix Distance matrix
 * @param current_path Current path
 * @param current_distance Distance of the current path
 * @param total_city Total number of cities
 * @param b Branch
 * @param best_distance Best distance
 * @param is_path_complete True if the path is complete, false otherwise
 * @param status MPI_Status
 * @param npes Number of processes
 * @param myrank Rank of the process
 * @param count Count the number of calls
 * @return True if the path is the better path, false otherwise
 */
bool isBestPath(int *distance_matrix, int *current_path, int current_distance, int total_city, struct Branch *b, int *best_distance, bool is_path_complete, MPI_Status status, int npes, int myrank, int *count){
    int recv_buffer;

    // Check if there is a new best distance from the other processes to recevie every 100 calls
    if (*count == 100) {
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
            MPI_Recv(&recv_buffer, 1, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Update the best distance of the process only if the received distance is better
            if (recv_buffer < *best_distance) {
                *best_distance = recv_buffer;
            }
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }
        *count = 0;
    }
    *count += 1;


    if (current_distance < *best_distance) {
        // If the path is complete
        if (is_path_complete) {
            recv_buffer = current_distance;
            // Create an array of MPI_Request objects to track the non-blocking sends
            MPI_Request requests[npes - 1];

            int req_count = 0; // counter for requests
            for (int i = 0; i < npes; i++) {
                if (i != myrank) {
                    // Send the message to process i
                    MPI_Isend(&recv_buffer, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_count]);
                    req_count++;
                }
            }
            requests[myrank] = MPI_REQUEST_NULL;

            // Wait for all the non-blocking sends to complete
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

            // Update the best distance
            *best_distance = current_distance;
            // Update the best path
            b->distance = current_distance;

        }
        return true;
    }

    return false;
}

/**
 * Determine the number of root path to have at least one path per process
 * and determine the size of the array to store all the root paths
 * For example, if there are 3 processes and 4 cities :
 * - The number of root path is 3
 * - The size of the array is 3 * 4 = 12
 * Modify the target_level to the level of the root_path, allowing to know where to start the branch and bound
 * @param total_city
 * @param nb_processor npes
 * @param size_array_final
 * @param target_level Level of the root_path
 * @return The size of the array to store all the root paths
 */
int getRootPathNumber(int total_city, int nb_processor, int size_array_final, int *target_level) {

    int node_number = 0;
    int level = 0;

    if (nb_processor == 1) {
        *target_level = level;
        return nb_processor * total_city;
    }

    while (nb_processor > node_number) {
        level += 1;
        if (level == 1) {
            node_number = (total_city - level);
        } else {
            node_number = (total_city - level) * (total_city - (level - 1));
        }
    };

    *target_level = level;
    size_array_final = node_number * total_city;
    return size_array_final;
}

/**
 * Generate all the root paths that will be used to start the branch and bound root_paths
 * Store all the root paths in an 1D array root_paths which will scatter between the processes
 * For example, if there are 3 processes and 4 cities :
 * - root_paths = [1, 2, 0, 0, 1, 3, 0, 0, 1, 4, 0, 0]
 * @param start_path Array containing a first path defining with the city to start (fixed to 1 for the tests)
 * @param total_city
 * @param level
 * @param target_level Level determine by the function getRootPathNumber
 * @param final_size_array Size determine by the function getRootPathNumber
 * @return Array containing all the root path
 */
int* generateRootPath(int *start_path, int total_city, int level, int target_level, int *final_size_array) {

    if (target_level == 0) {
        return start_path;
    }

    int current_node_number;

    if (level == 0) {
        current_node_number = 1;
    } else if (level == 1) {
        current_node_number = (total_city - level);
    } else {
        current_node_number = (total_city - level) * (total_city - (level - 1));
    }

    int *root_paths;

    root_paths = malloc(sizeof(int *) * *final_size_array);

    int jump = 0;

    for (int p_n = 0; p_n < current_node_number; p_n += 1) {
        int path[total_city];

        for (int i = 0; i < total_city; ++i) {
            path[i] = start_path[p_n * total_city +i ];
        }

        int path_temp[total_city];

        for (int j = 0; j < total_city; ++j) {
            for (int i = 0; i < total_city; ++i) {
                path_temp[i] = path[i];
            }
            path_temp[level + 1] = j + 1;
            if (isPathValid(path_temp, total_city)){
                for (int i = 0; i < total_city; ++i) {
                    root_paths[i + jump] = path_temp[i];
                }
                jump += total_city;
            }

        }

    }
    if (level < target_level - 1) {
        generateRootPath(root_paths, total_city, level + 1, target_level, final_size_array);
    } else {
        return root_paths;
    }
}

/**
 * Initailize the visited_city array of boolean allowing to know if a city has already been visited
 * - For example, if the path is [1, 2, 4, 0] and the level is 2, the visited_city array will be [true, true, false, true]
 * @param path
 * @param total_city
 * @param level
 * @return Array of boolean
 */
bool* initVisitedCity(int* path, int total_city, int level) {
    bool *visited_city = malloc(total_city * sizeof(bool));

    for (int i = 0; i < total_city; i++) {
        visited_city[i] = false;
    }

    for (int i = 0; i <= level; ++i) {
        visited_city[path[i]-1] = true;
    }

    return visited_city;
}

/**
 * Main function of the branch and bound algorithm
 * Update the path of the best branch if a better path is found
 * @param distance_matrix
 * @param visited_city
 * @param total_city
 * @param current_path
 * @param current_distance
 * @param b
 * @param best_distance
 * @param level
 * @param status
 * @param npes
 * @param myrank
 * @param count
 * @return True if the path is valid, false otherwise
 */
void doBnBWsp(int *distance_matrix, bool* visited_city, int total_city, int *current_path, int current_distance, struct Branch *b, int *best_distance, int level, MPI_Status status, int npes, int myrank, int* count){

    if (level < total_city) {
        int current_distance_temp = current_distance;

        for (int i = 0; i < total_city; ++i) {
            current_path[level] = i + 1;

            if (!visited_city[current_path[level] - 1]) {
                current_distance += distance_matrix[((current_path[level - 1] - 1) * total_city) + (current_path[level] - 1)];

                if (isBestPath(distance_matrix, current_path, current_distance, total_city, b, best_distance, false, status, npes, myrank, count)) {
                    visited_city[current_path[level] - 1] = true;
                    doBnBWsp(distance_matrix, visited_city, total_city, current_path, current_distance, b, best_distance, level + 1, status, npes, myrank, count);
                    visited_city[current_path[level] - 1] = false;
                }
                current_distance = current_distance_temp;
            }
            current_path[level] = 0;
        }
    } else {
        if (isBestPath(distance_matrix, current_path, current_distance, total_city, b, best_distance, true, status, npes, myrank, count)){
            memcpy(b->path, current_path, sizeof(int) * total_city);
        }
    }
}




int main(int argc, char *argv[])
{
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;

    int npes, myrank;
    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    t1 = MPI_Wtime();
    t3 = MPI_Wtime();

    char *file_name = argv[1];

    //--------------------------- Distance Matrix initialisation ------------------------ //
    int total_city;
    int *distance_matrix;

    // Only process 0 reads the file and determines the total number of cities
    if (myrank == 0) {
        total_city = getTotalCity(file_name);
    }

    // Broadcast the total number of cities to all processes
    MPI_Bcast(&total_city, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process allocates the memory for the distance matrix
    distance_matrix = malloc((total_city * total_city) * sizeof(int));

    // Only process 0 reads the file and fills the distance matrix
    if (myrank == 0) {
        distance_matrix = createDistanceMatrix(file_name, total_city);
    }

    // Broadcast the distance matrix to all processes
    MPI_Bcast(distance_matrix, (total_city * total_city), MPI_INT, 0, MPI_COMM_WORLD);

    //--------------------------- Init first branch ------------------------ //

    struct Branch best_branch;

    // Each process allocates the memory for the best branch and init the distance to the max value
    best_branch.path = malloc(total_city * sizeof(int));
    best_branch.distance = INT_MAX;

    //------------------------------- Branch and Bound ------------------------------------//

    int *root_paths;
    int target_level;
    int final_size_array;

    // Only process 0
    if (myrank == 0) {
        // Get the number of root paths to generate
        final_size_array = getRootPathNumber(total_city, npes, final_size_array, &target_level);

        // Allocate the memory for the array containing all the root paths
        root_paths = malloc(sizeof(int *) * final_size_array);

        // Init a first path of the beginning city
        int *path;
        path = malloc(sizeof(int *) * total_city);
        memset(path, 0, total_city * sizeof(int));
        path[0] = 1;

        int start_level = 0;

        // Generate all the root paths and store them in the root_paths array
        root_paths = generateRootPath(path, total_city, start_level, target_level, &final_size_array);
    }

     // Broadcast the target level to all processes
    MPI_Bcast(&target_level, 1, MPI_INT, 0, MPI_COMM_WORLD );
    // Broadcast the final size of the root_paths array to all processes
    MPI_Bcast(&final_size_array, 1, MPI_INT, 0, MPI_COMM_WORLD );

    // Get the number of root path to generate
    int total_root_node = final_size_array / total_city;

    // Init a minimum number of root path to allocate per process
    int process_root_node = (int)floor(total_root_node / npes);

    // Init an array containing the number of root path to allocate per process
    int sendcounts[npes];
    for (int i = 0; i < npes; i++) {
        sendcounts[i] = process_root_node * total_city;
    }

    // If the number of root path to generate is not a multiple of the number of processes
    // We add the remaining root path one by one to each process starting from process 0
    if (total_root_node % npes != 0) {
        if (npes > total_root_node % npes) {
            for (int i = 0; i < total_root_node % npes; ++i) {
                sendcounts[i] += total_city;
            }
        }
    }

    // Init of displs array for the scatterv function
    int displs[npes];
    for (int i = 0; i < npes; i++) {
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    // Each process init an array of the size of the number of root path that it will receive
    int *process_root_paths;
    process_root_paths = malloc(sendcounts[myrank] * total_city * sizeof(int));

    // Scatter the root_paths array to all processes
    MPI_Scatterv(root_paths, sendcounts, displs, MPI_INT, process_root_paths, sendcounts[myrank], MPI_INT, 0, MPI_COMM_WORLD);

    t4 = MPI_Wtime();
    t5 = MPI_Wtime();

    int best_distance = best_branch.distance;
    int *root_path_start = malloc(total_city * sizeof(int));
    bool* visited_city;

    // For each root path received, each process will compute the best branch
    for (int i = 0; i < sendcounts[myrank] / total_city; ++i) {
        // Init the root path from the process_root_paths array
        for (int j = 0; j < total_city; ++j) {
            root_path_start[j] = process_root_paths[i * total_city + j];
        }

        // Init the visited city array
        visited_city = initVisitedCity(root_path_start, total_city, target_level);

        // Init the current distance from the root_path_start
        int current_distance = 0;
        for (int j = 0; j < target_level; ++j) {
            current_distance += distance_matrix[((root_path_start[j] - 1) * total_city) + (root_path_start[j + 1] - 1)];
        }

        // Init a count
        int count = 0;

        // Do the branch and bound
        doBnBWsp(distance_matrix, visited_city, total_city, root_path_start, current_distance, &best_branch, &best_distance, target_level + 1, status, npes, myrank, &count);
    }

    t6 = MPI_Wtime();

    //------------------------------- Gather the best branch ------------------------------------//

    // Init two structure to store the local and global distance and rank
    struct {
        int distance;
        int rank;
    } best_local_data, best_global_data;

    // Init the local
    best_local_data.distance = best_branch.distance;
    best_local_data.rank = myrank;

    t7 = MPI_Wtime();

    // Get the best distance from all processes and the rank of the process that found it
    MPI_Allreduce(&best_local_data, &best_global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    t8 = MPI_Wtime();

    t9 = MPI_Wtime();

    // If the best distance is not found by process 0, we send the best path to process 0
    if (best_global_data.rank != 0) {
        if (myrank == 0){
            MPI_Recv(best_branch.path, total_city, MPI_INT, best_global_data.rank, 1, MPI_COMM_WORLD, &status);
        } else if (myrank == best_global_data.rank) {
            MPI_Send(best_branch.path, total_city, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    t10 = MPI_Wtime();

    if (myrank == 0) {
        printf("Best Path: ");
        for (int i = 0; i < total_city; i++) {
            printf("%d ", best_branch.path[i]);
        }
        printf("\n");

        printf("Best distance is equal to: %d\n",  best_global_data.distance);

        printf("\n--------------------------------------------------------------\n");
    }


    t11 = MPI_Wtime();

    // Clean the potential message waiting for reception before exiting the program
    int flag = 0;
    int recv_buffer;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status );
    while (flag){
        MPI_Recv(&recv_buffer, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status );
    }

    t12 = MPI_Wtime();
    t2 = MPI_Wtime();


    if (myrank == 0) {
        printf("Scatter time: %1.4f\n", t4-t3);
        printf("Branch and bound time: %1.4f\n", t6-t5);
        printf("All reduce time: %1.4f\n", t8-t7);
        printf("Send best time: %1.4f\n", t10-t9);
        printf("Clean time: %1.4f\n", t12-t11);
        printf("Total time: %1.4f\n", t2-t1);
    }

    free(distance_matrix);
    free(best_branch.path);

    MPI_Finalize();
}




