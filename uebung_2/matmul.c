#include "matmul.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>


void matmul(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

void matmul_swapped_loops(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}


void matmul_unrolled_loops(double *A, double *B, double *C, int n) {
    int unrollsize = 2; //2,4,8,16,32,64
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j += unrollsize) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
                C[i * n + j + 1] += A[i * n + k] * B[k * n + j + 1];
            }
        }
    }
}

void matmul_blocking(double *A, double *B, double *C, int n) {
    int tilesize = 2; //2,4,8,16,32,64
    int unrollsize = 2; //2,4,8,16,32,64
    for (int i1 = 0; i1 < n; i1 += tilesize) {
        for (int k1 = 0; k1 < n; k1 += tilesize) {
            for (int i2 = 0; i2 < tilesize; i2++) {
                for (int k2 = 0; k2 < tilesize; k2++) {
                    for (int j = 0; j < n; j += unrollsize) {
                        C[(i1 + i2) * n + j] += A[(i1 + i2) * n + (k1 + k2)] * B[(k1 + k2) * n + j];
                        C[(i1 + i2) * n + j + 1] += A[(i1 + i2) * n + (k1 + k2)] * B[(k1 + k2) * n + j + 1];
                    }
                }
            }
        }
    }
}

void matmul_pointer(double *A, double *B, double *C, int n) {
    /*for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j += 2) { //2,4,8,16,32,64
                *(C + i * n + j) += *(A + i * n + k) * *(B + k * n + j);
                *(C + i * n + j + 1) += *(A + i * n + k) * *(B + k * n + j + 1);
            }
        }
    }*/
    int tilesize = 2; //2,4,8,16,32,64
    int unrollsize = 2; //2,4,8,16,32,64
    for (int i1 = 0; i1 < n; i1 += tilesize) {
        for (int k1 = 0; k1 < n; k1 += tilesize) {
            for (int i2 = 0; i2 < tilesize; i2++) {
                for (int k2 = 0; k2 < tilesize; k2++) {
                    for (int j = 0; j < n; j += unrollsize) {
                        *(C + (i1 + i2) * n + j) += *(A + (i1 + i2) * n + (k1 + k2)) * *(B + (k1 + k2) * n + j);
                        *(C + (i1 + i2) * n + j + 1) += *(A + (i1 + i2) * n + (k1 + k2)) * *(B + (k1 + k2) * n + j + 1);
                    }
                }
            }
        }
    }
}

void matmul_constants(double *A, double *B, double *C, int n) {
    /*for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double temp_a = *(A + i * n + k)
            for (int j = 0; j < n; j++)
                *(C + i * n + j + 1) += temp_a * *(B + k * n + j + 1);
        }*/
    int tilesize = 2; //2,4,8,16,32,64
    int unrollsize = 2; //2,4,8,16,32,64
    for (int i1 = 0; i1 < n; i1 += tilesize) {
        for (int k1 = 0; k1 < n; k1 += tilesize) {
            for (int i2 = 0; i2 < tilesize; i2++) {
                for (int k2 = 0; k2 < tilesize; k2++) {
                    double temp_a = *(A + (i1 + i2) * n + (k1 + k2));
                    for (int j = 0; j < n; j += unrollsize) {
                        *(C + (i1 + i2) * n + j) += temp_a * *(B + (k1 + k2) * n + j);
                        *(C + (i1 + i2) * n + j + 1) += temp_a * *(B + (k1 + k2) * n + j + 1);
                    }
                }
            }
        }
    }
}

int results_correct(double* A, double* B, int n){
    int correct = 1;
    for(int i = 0; i<n*n; i++){
        if(*(A+i) != *(B+i)){
            correct = 0;
            break;
        }
    }
    return correct;
}


int main(int args, char *argsv[]) {
    if (args != 2) {
        printf("Please specify matrix size.\n");
        return 0;
    }

    int n;
    sscanf(argsv[1], "%d", &n);

    srand(time(NULL));

    double *ptr_A = (double *) malloc(n * n * sizeof(double));
    double *ptr_B = (double *) malloc(n * n * sizeof(double));
    double *ptr_C = (double *) malloc(n * n * sizeof(double));
    double *ptr_C_ref = (double *) malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ptr_A[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_B[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_C[i * n + j] = (double) 0;
            ptr_C_ref[i * n + j] = (double) 0;
        }
    }

    float runtimes[5][10];
    for (int m = 0; m < 10; m++) {
        // Time start
        struct timeval tv1, tv2;

        gettimeofday(&tv1, NULL);
        matmul(ptr_A, ptr_B, ptr_C_ref, n);
        gettimeofday(&tv2, NULL);
        runtimes[0][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);

        gettimeofday(&tv1, NULL);
        matmul_swapped_loops(ptr_A, ptr_B, ptr_C, n);
        gettimeofday(&tv2, NULL);
        runtimes[1][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if(m==0 && n==16){
            if(results_correct(ptr_C_ref, ptr_C, n)) printf("Optimierung 1: Ergebnis korrekt\n");
            else printf("Optimierung 1: Ergebnis nicht korrekt\n");
        }

        gettimeofday(&tv1, NULL);
        matmul_unrolled_loops(ptr_A, ptr_B, ptr_C, n);
        gettimeofday(&tv2, NULL);
        runtimes[2][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if(m==0 && n==16){
            if(results_correct(ptr_C_ref, ptr_C, n)) printf("Optimierung 2: Ergebnis korrekt\n");
            else printf("Optimierung 2: Ergebnis nicht korrekt\n");
        }

        gettimeofday(&tv1, NULL);
        matmul_blocking(ptr_A, ptr_B, ptr_C, n);
        gettimeofday(&tv2, NULL);
        runtimes[3][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if(m==0 && n==16){
            if(results_correct(ptr_C_ref, ptr_C, n)) printf("Optimierung 3: Ergebnis korrekt\n");
            else printf("Optimierung 3: Ergebnis nicht korrekt\n");
        }

        gettimeofday(&tv1, NULL);
        matmul_pointer(ptr_A, ptr_B, ptr_C, n);
        gettimeofday(&tv2, NULL);
        runtimes[3][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if(m==0 && n==16){
            if(results_correct(ptr_C_ref, ptr_C, n)) printf("Optimierung 4: Ergebnis korrekt\n");
            else printf("Optimierung 4: Ergebnis nicht korrekt\n");
        }

        gettimeofday(&tv1, NULL);
        matmul_constants(ptr_A, ptr_B, ptr_C, n);
        gettimeofday(&tv2, NULL);
        runtimes[4][m] = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        if(m==0 && n==16){
            if(results_correct(ptr_C_ref, ptr_C, n)) printf("Optimierung 5: Ergebnis korrekt\n");
            else printf("Optimierung 5: Ergebnis nicht korrekt\n");
        }

    }

    for (int i = 0; i < 5; i++) {
        printf("Variante %i:   ", i);
        for (int j = 0; j < 10; j++) {
            printf("%f   ", runtimes[i][j]);
        }
        printf("\n");
    }

    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
    free(ptr_C_ref);

    return 0;
}
