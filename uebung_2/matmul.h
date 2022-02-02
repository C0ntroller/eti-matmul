
#ifndef UE_MATMUL_H
#define UE_MATMUL_H

void matmul(double* A, double* B, double* C, int n);

void matmul_swapped_loops(double* A, double* B, double* C, int n);

void matmul_unrolled_loops(double* A, double* B, double* C, int n);

void matmul_blocking(double* A, double* B, double* C, int n);

void matmul_pointer(double* A, double* B, double* C, int n);

void matmul_constants(double* A, double* B, double* C, int n);

void matmul_omp(double* A, double* B, double* C, int n);


#endif //UE_MATMUL_H
