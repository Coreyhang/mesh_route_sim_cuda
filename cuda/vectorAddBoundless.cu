#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void vectorAddBoundless(int *a, int *b, int *c, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < N) {
        c[tid] = a[tid] + c[tid];
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + c[tid];
}

int main(void) {
    int N = 65536 * 1024 * 16;
    size_t size = N * sizeof(int);

    int *a = (int *)malloc(size);
    int *b = (int *)malloc(size);
    int *c = (int *)malloc(size);

    // 初始化输入向量A和B
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    clock_t start_c = clock();
    for (int i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
    clock_t stop_c = clock();


    // Device内存分配
    int *dev_a = NULL;
    int *dev_b = NULL;
    int *dev_c = NULL;
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // 将Host端的数据复制到Device端
    printf("Copy input data from the host memory to the CUDA device\n");
    clock_t start_cpy = clock();
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    clock_t stop_cpy = clock();

    // 调用CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    clock_t start_cuda = clock();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    clock_t stop_cuda = clock();

    threadsPerBlock = 128;
    blocksPerGrid = 128;
    clock_t start_cuda_b = clock();
    vectorAddBoundless<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    clock_t stop_cuda_b = clock();

    // 将结果从Device端复制到Host端
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    float time_c = (float)(stop_c - start_c);
    float time_cuda = (float)(stop_cuda - start_cuda);
    float time_cuda_b = (float)(stop_cuda_b - start_cuda_b);
    float time_cpy = (float)(stop_cpy - start_cpy);

    printf("Computation time of C is %f, computation time of CUDA is %f\n", time_c, time_cuda);
    printf("Computation time of CUDA is %f\n", time_cuda_b);
    time_cuda += time_cpy;
    printf("Computation time of CUDA including the memory copy time is %f\n", time_cuda);

    // 释放Device内存和Host内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    printf("Done\n");
    return 0;
}