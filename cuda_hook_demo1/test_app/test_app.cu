#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 启动内核
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 修改后的内核启动语法
    cudaStream_t stream;
    cudaStreamCreate(&stream);  // 创建新的CUDA流

    // 原始启动方式保持
    // 以这个方式启动不一定会调用 cudaLaunchKernel,实验证明，这个函数会调用cudaLaunch 而不是cudaLaunchKernel
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, N);

    vectorAdd<<<grid, block,1024,stream>>>(d_a, d_b, d_c, N);
    
    // 修改后的显式cudaLaunchKernel调用
    int arg_n = N;  // 创建非const变量
    void* args[] = {
        &d_a, 
        &d_b, 
        &d_c, 
        &arg_n  // 使用非const变量的地址
    };

    cudaLaunchKernel(
        (void*)vectorAdd,
        grid, block,
        args,
        0,
        nullptr
    );
    
    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}