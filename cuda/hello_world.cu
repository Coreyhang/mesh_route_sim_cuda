#include "stdio.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    cudaDeviceProp prop;
    int count;
    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to count cuda devices (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Number of CUDA devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get cuda properties (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        printf("---General Information for device %d---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Total Global Memory: %ld\n", prop.totalGlobalMem);
        printf("Total Constant Memory: %ld\n", prop.totalConstMem);
        printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("Shared Memory per Multiprocessor: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per Multiprocessor: %d\n", prop.regsPerBlock);
        printf("Threads in Warp: %d\n", prop.warpSize);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Thread Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    int dev;
    cudaGetDevice(&dev);  // 查看当前使用的CUDA设备
    printf("ID of current CUDA device: %d\n", dev);

    // 将某些属性填充到一个cudaDeviceProp结构中
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;

    cudaChooseDevice(&dev, &prop);  // 返回满足条件的一个设备ID
    cudaSetDevice(dev); // 设置CUDA设备
    
    
    int c;
    int *dev_c;
    size_t size = sizeof(int);
    err = cudaMalloc((void **)&dev_c, size);  // 在设备上分配内存

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device number c (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    add<<<1, 1>>>(2, 7, dev_c);  // NVCC编译, <<<>>>中的参数告诉runtime如何启动设备代码, ()中为kernel函数参数

    err = cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy number c from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("2 + 7 = %d\n", c);

    printf("Hello World!\n");

    cudaFree(dev_c);
}