#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

void main(int args, char* argsv[]) {
    if (args != 2) {
        printf("Please specify matrix size.\n");
        return;
    }

    int n;
    sscanf(argsv[1], "%d", &n);

    srand(time(NULL));

    double* ptr_A = (double*) malloc(n*n*sizeof(double));
    double* ptr_B = (double*) malloc(n*n*sizeof(double));
    double* ptr_C = (double*) malloc(n*n*sizeof(double));

    int i, j, k, m;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            ptr_A[i*n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_B[i*n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            ptr_C[i*n + j] = (double) 0;
        }
    }

    for (m = 0; m < 10; m++) {
        // Time start
        struct timeval tv1, tv2;
        gettimeofday(&tv1, NULL);

        // Mat mul
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                for (k = 0; k < n; k++)
                    ptr_C[i*n + j] += ptr_A[i*n + k] * ptr_B[k*n + j];

        // Time end
        gettimeofday(&tv2, NULL);
        // print timeend - timestart
        printf ("%f\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + 1000 * (double) (tv2.tv_sec - tv1.tv_sec));
    }

    free(ptr_A);
    free(ptr_B);
    free(ptr_C);
}
