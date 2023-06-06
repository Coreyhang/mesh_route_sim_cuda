#include <stdio.h>
#include <cuda_runtime.h>

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
const int threadsPerBlock = 256;

__global__ void dotProduct(double *a, double *b, double *c, int N) {
    __shared__ double cache[threadsPerBlock];  // 根据线程块中线程的数量来分配共享内存
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;  // 共享内存中的偏移等于线程索引
    double temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp; // 将temp写入到共享缓冲区中

    // 接下来需要对cache中的值求和
    // 需要某种方法保证读取cache在所有并行的写入操作之后
    __syncthreads();  // 对线程块中的线程进行同步
    // 确保线程块中的所有线程都执行完前面的语句后才会执行后面的语句

    // 并行规约运行
    // 算法复杂度为O(log2N)
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i = i / 2;
    }

    if (cacheIndex == 0)  // 相当于选择了每个线程块中的线程0来执行下面的操作
        c[blockIdx.x] = cache[0];  // 去掉if后每个线程都会执行此操作
}

int main(void) {
    int numElements = 65536;
    int temp = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = 32 < temp ? 32 : temp;
    size_t sizeInput = numElements * sizeof(double);
    size_t sizeResult = blocksPerGrid * sizeof(double);

    double *a, *b, c, *partial_c;
    double *dev_a, *dev_b, *partial_dev_c;

    // CPU上分配内存
    a = new double[numElements];
    b = new double[numElements];
    partial_c = new double[blocksPerGrid];

    // GPU上分配内存
    cudaMalloc((void **)&dev_a, sizeInput);
    cudaMalloc((void **)&dev_b, sizeInput);
    cudaMalloc((void **)&partial_dev_c, sizeResult);

    // 初始化输入向量
    for (int i = 0; i < numElements; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // double *e = new double[numElements];
    // printf("%d, %d\n", (int)sizeof(double), (int)sizeof(char));

    // 将数据从CPU拷贝到GPU
    cudaMemcpy(dev_a, a, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeInput, cudaMemcpyHostToDevice);
    
    // cudaMemcpy(e, dev_a, sizeInput, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < numElements; ++i) {
    //     if (a[i] - e[i] > 1e-5) {
    //         printf("!");
    //     }
    // }


    // 执行CUDA Kernel
    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, partial_dev_c, numElements);

    // 将结果从GPU拷贝到CPU
    cudaMemcpy(partial_c, partial_dev_c, sizeResult, cudaMemcpyDeviceToHost);

    // 在CPU上完成最终的计算
    c = 0;
    for (int i = 0; i < blocksPerGrid; ++i)
        c += partial_c[i];

    // Check
    double truth = 2 * sum_squares((double)(numElements - 1));
    printf("The result is %lf and the ground truth is %lf\n", c, truth);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(partial_dev_c);
    delete [] a;
    delete [] b;
    delete [] partial_c;
}