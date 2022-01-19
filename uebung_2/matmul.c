#include "matmul.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>


void matmul(double* A, double* B, double* C, int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void matmul_swapped_loops(double* A, double* B, double* C, int n){
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void matmul_constants(double* A, double* B, double* C, int n){
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            double temp_a = A[i*n + k]
            for (int j = 0; j < n; j++)
                C[i*n + j] += temp_a * B[k*n + j];
}

void matmul_unrolled_loops(double* A, double* B, double* C, int n){
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j += 2) { //2,4,8,16,32,64
                C[i * n + j] += A[i * n + k] * B[k * n + j];
                C[i * n + j + 1] += A[i * n + k] * B[k * n + j + 1];
            }
        }
    }
}

void matmul_pointer(double* A, double* B, double* C, int n){
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j += 2) { //2,4,8,16,32,64
                C[i * n + j] += A[i * n + k] * B[k * n + j];
                *(C + i * n + j + 1) += *(A + i * n + k) * *(B + k * n + j + 1);
            }
        }
    }
}


int main(int args, char* argsv[]) {
    if (args != 2) {
        printf("Please specify matrix size.\n");
        return 0;
    }

    int n;
    sscanf(argsv[1], "%d", &n);

    srand(time(NULL));

    double* ptr_A = (double*) malloc(n*n*sizeof(double));
    double* ptr_B = (double*) malloc(n*n*sizeof(double));
    double* ptr_C = (double*) malloc(n*n*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ptr_A[i*n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_B[i*n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_C[i*n + j] = (double) 0;
        }
    }

    for (int m = 0; m < 10; m++) {
        // Time start
        struct timeval tv1, tv2;
        gettimeofday(&tv1, NULL);

        // Mat mul
        matmul(ptr_A, ptr_B, ptr_C, n);

        // Time end
        gettimeofday(&tv2, NULL);
        // print timeend - timestart
        printf ("%f\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec));
    }

    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
}
