#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 4
#define HANDLE_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))

static void handleCudaError(cudaError_t err, const char *file, int line){
    if(err!=cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void matmul_cpu(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

int results_correct(double* A, double* B, int n){
    for(int i = 0; i<n*n; i++){
        if((*(A+i) - *(B+i)) > 0.0001){
            return 0;
        }
    }
    return 1;
}

__global__ void matmul_gpu(double *A, double *B, double *C, int n){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n; j += blockDim.y * gridDim.y) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Please specify matrix size.\n");
        return 0;
    }

    int n;
    sscanf(argv[1], "%d", &n);

    // Allocate and initialize host memory
    double *C_check = (double *) malloc(n * n * sizeof(double));

    // Allocate cuda-managed memory
    double *A, *B, *C;
    HANDLE_ERROR(cudaMallocManaged(&A, n*n*sizeof(double)));
    HANDLE_ERROR(cudaMallocManaged(&B, n*n*sizeof(double)));
    HANDLE_ERROR(cudaMallocManaged(&C, n*n*sizeof(double)));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            B[i * n + j] = (double) ((rand() % 2000) + 9000) / 10000;
            C[i * n + j] = (double) 0;
            C_check[i * n + j] = (double) 0;
        }
    }

#if 0
    // Run reference implementation on cpu
    matmul_cpu(A, B, C_check, n);
#endif


    // Get device id
    int deviceId;
    HANDLE_ERROR(cudaGetDevice(&deviceId));


    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(ceil(n/BLOCK_SIZE), ceil(n/BLOCK_SIZE));

    // Prefetch data to gpu
    HANDLE_ERROR(cudaMemPrefetchAsync(A, n*n*sizeof(double), deviceId));
    HANDLE_ERROR(cudaMemPrefetchAsync(B, n*n*sizeof(double), deviceId));
    HANDLE_ERROR(cudaMemPrefetchAsync(C, n*n*sizeof(double), deviceId));

    cudaEvent_t cstart, cend;
    cudaStream_t cstream;
    HANDLE_ERROR(cudaEventCreate(&cstart));
    HANDLE_ERROR(cudaEventCreate(&cend));
    HANDLE_ERROR(cudaStreamCreate(&cstream));
    float runtime = 0;

    for(int i = 0; i<10; i++) {

        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaEventRecord(cstart, cstream));

        // Perform matmul on n elements
        matmul_gpu<<<gridSize, blockSize>>>(A, B, C, n);
        HANDLE_ERROR(cudaGetLastError());

        // Prefetch data from gpu
        HANDLE_ERROR(cudaMemPrefetchAsync(C, n * n * sizeof(double), cudaCpuDeviceId));

        HANDLE_ERROR(cudaEventRecord(cend, cstream));
        HANDLE_ERROR(cudaEventSynchronize(cend));
        HANDLE_ERROR(cudaEventElapsedTime(&runtime, cstart, cend));
        printf("%f,", runtime);

        HANDLE_ERROR(cudaMemset(C, 0, n * n * sizeof(double)));
    }

    printf("\n");

#if 0
    if(!results_correct(C, C_check, n)){
        printf("incorrect results\n");

        HANDLE_ERROR(cudaFree(A));
        HANDLE_ERROR(cudaFree(B));
        HANDLE_ERROR(cudaFree(C));

        free(C_check);

        return 1;
    }
#endif

    HANDLE_ERROR(cudaFree(A));
    HANDLE_ERROR(cudaFree(B));
    HANDLE_ERROR(cudaFree(C));

    free(C_check);

    return 0;
}