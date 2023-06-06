#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>


// CUDA Kernel Device code
__global__ void MatAdd(float *A, float *B, float *C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

// Host code
int main(void) {
    // 记录时间用
    clock_t start_c, stop_c, start_cuda, stop_cuda;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int numRows = 10000;
    int numCols = 10000;
    printf("Matrix addition of %d * %d elements\n", numRows, numCols);
    size_t size = numCols * numCols * sizeof(float);

    // 初始化输入矩阵A和B
    float **A = (float **)malloc(sizeof(float *) * numRows);
    float **B = (float **)malloc(sizeof(float *) * numRows);
    float **C = (float **)malloc(sizeof(float *) * numRows);
    for (int i = 0; i < numRows; ++i) {
        A[i] = (float *)malloc(sizeof(float) * numCols);
        B[i] = (float *)malloc(sizeof(float) * numCols);
        C[i] = (float *)malloc(sizeof(float) * numCols);
    }
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            A[i][j] = rand() / (float)RAND_MAX;
            B[i][j] = rand() / (float)RAND_MAX;
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    
    start_c = clock();
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    stop_c = clock();

    // Host内存分配
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            h_A[j + i * numCols] = A[i][j];
            h_B[j + i * numCols] = B[i][j];
        }
    }

    // Device内存分配
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 将Host端的数据复制到Device端
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Host调用vectorAdd CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows * numCols + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    start_cuda = clock();
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    stop_cuda = clock();

    float time_c = (float)(stop_c - start_c);
    float time_cuda = (float)(stop_cuda - start_cuda);

    printf("Computation time of C is %f and computation time of CUDA is %f\n", time_c, time_cuda);

    // 将结果从Device端复制到Host端
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            if (fabs(C[i][j] - h_C[j + i * numCols]) > 1e-5) {
                fprintf(stderr, "Result verification failed at element (%d, %d), truth is %f while get %f\n", i, j, C[i][j], h_C[j + i * numCols]);
                exit(EXIT_FAILURE);
            }
        }
    }

    // 释放Device内存和Host内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}