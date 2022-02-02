#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

void matmul_swapped_loops(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
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
    if (args != 2) {
        printf("Please specify matrix size.\n");
        return 0;
    }

    int n;
    sscanf(argsv[1], "%d", &n);
    srand(time(NULL));

    double *ptr_A, *ptr_B, *ptr_C; //, *ptr_C_check;
    ptr_B = (double *) malloc(n * n * sizeof(double));
    ptr_A = (double *) malloc(n * n * sizeof(double));
    ptr_C = (double *) malloc(n * n * sizeof(double));
    //ptr_C_check = (double *) malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ptr_A[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_B[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
        }
    }
    memset(ptr_C, 0, n * n * sizeof(double));
    //memset(ptr_C_check, 0, n * n * sizeof(double));

    double runtime;
    struct timeval tv1, tv2;

    //matmul_swapped_loops(ptr_A, ptr_B, ptr_C_check, n);

    for (int i = 0; i < 10; i++) {
        gettimeofday(&tv1, NULL);
        #pragma omp target data map(to: ptr_A[:n*n]) map(to: ptr_B[:n*n]) map(tofrom: ptr_C[:n*n])
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        ptr_C[i * n + j] += ptr_A[i * n + k] * ptr_B[k * n + j];
        }
        #pragma omp barrier

        gettimeofday(&tv2, NULL);

        /*if (!results_correct(ptr_C, ptr_C_check, n)) {
            printf("Incorrect\n");
            for (int i = 0; i < n; i++) {
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
            }

            return 1;
        }*/

        runtime = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec);
        printf("%f, ", runtime);

        memset(ptr_C, 0, n * n * sizeof(double));
    }

    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
    //free(ptr_C_check);

    printf("\n");
    return 0;
}
