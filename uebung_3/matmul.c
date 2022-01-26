#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void matmul_swapped_loops(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

void matmul_mpi(double* A, double* B, double* C, int rows, int n) {

    for (int i = 0; i < rows; i++)
        for (int k = 0; k < n; k++) {
            double a_tmp = *(A + i * n + k);
            for (int j = 0; j < n; j++)
                *(C + i * n + j) += a_tmp * *(B + k * n + j);
        }
}

int results_correct(double* A, double* B, int n){
    for(int i = 0; i<n*n; i++){
        if(*(A+i) != *(B+i)){
            return 0;
        }
    }
    return 1;
}


int main(int args, char *argsv[]) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (args != 2) {
        if (rank == 0) printf("Please specify matrix size.\n");
        return 0;
    }

    int n;
    sscanf(argsv[1], "%d", &n);
    srand(time(NULL));

    int rows = (int) n / size;
    int bfrsize = n * rows;

    int* slices = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        slices[i] = bfrsize;
    } 

    int* offsets = (int *) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        offsets[i] = bfrsize * i;
    }

    double *ptr_A, *ptr_B, *ptr_C, *ptr_A_tile, *ptr_C_result, *ptr_C_check;
    ptr_B = (double *) malloc(n * n * sizeof(double));

    if (rank == 0) {
        ptr_A = (double *) malloc(n * n * sizeof(double));
        ptr_C = (double *) malloc(n * n * sizeof(double));
        ptr_C_check = (double *) malloc(n * n * sizeof(double));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ptr_A[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
                ptr_B[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
                ptr_C[i * n + j] = (double) 0;
                ptr_C_check[i * n + j] = (double) 0;
            }
        }
    }
    
    ptr_A_tile = (double *) malloc(bfrsize * sizeof(double));
    //ptr_B_tile = (double *) malloc(bfrsize * sizeof(double));
    ptr_C_result = (double *) malloc(bfrsize * sizeof(double));

    double runtime;
    struct timeval tv1, tv2;

    if(rank == 0) matmul_swapped_loops(ptr_A, ptr_B, ptr_C_check, n);

    for (int i = 0; i < 10; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tv1, NULL);
        MPI_Scatterv(ptr_A, slices, offsets, MPI_DOUBLE, ptr_A_tile, bfrsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ptr_B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        matmul_mpi(ptr_A_tile, ptr_B, ptr_C_result, rows, n);
        MPI_Gatherv(ptr_C_result, bfrsize, MPI_DOUBLE, ptr_C, slices, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank == 0 && n - (rows * size) != 0) {
            matmul_mpi(ptr_A + (n * rows * size), ptr_B, ptr_C + (n * rows * size), n % size, n);
        }
        gettimeofday(&tv2, NULL);

        if (rank == 0 && !results_correct(ptr_C, ptr_C_check, n)) {
            printf("Incorrect\n");
            /*for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%f\t", ptr_C[n*i + j]);
                }
                printf("\n");
            }

            printf("\n");
            printf("\n");

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%f\t", ptr_C_check[n*i + j]);
                }
                printf("\n");
            }*/

            return 1;
        }

        runtime = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if (rank == 0) printf("%f,", runtime);

        memset(ptr_C_result, 0, bfrsize * sizeof(double));
        if (rank == 0) memset(ptr_C, 0, n*n * sizeof(double));
    }

    if (rank == 0) {
        free(ptr_A);
        //free(ptr_B);
        free(ptr_C);
        free(ptr_C_check);
    }
    free(ptr_A_tile);
    free(ptr_B);
    free(ptr_C_result);

    MPI_Finalize();
    if(rank==0) printf("\n");
    return 0;
}
