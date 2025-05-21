#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>

#define NUM_CONTEXTS 25  // 增加上下文数量
#define ITERATIONS 1000000  // 增加迭代次数提高测量精度

// 添加内核函数
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Testing on %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // 创建多个上下文（带错误检查）
    CUcontext contexts[NUM_CONTEXTS];
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        CUresult err = cuCtxCreate(&contexts[i], 0, 0);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "创建上下文 %d 失败: %d\n", i, err);
            return 1;
        }
    }

    // 为每个上下文创建资源
    const int data_size = 1024*1024*32;
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        cuCtxSetCurrent(contexts[i]);
        
        // 分配设备内存
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, data_size * sizeof(float));
        cudaMalloc(&d_b, data_size * sizeof(float));
        cudaMalloc(&d_c, data_size * sizeof(float));

        // 分配并初始化主机内存
        float *h_a = new float[data_size];
        float *h_b = new float[data_size];
        for (int j = 0; j < data_size; j++) {
            h_a[j] = static_cast<float>(i);  // 不同上下文使用不同初始值
            h_b[j] = static_cast<float>(j % 100);
        }

        // 拷贝数据到设备
        cudaMemcpy(d_a, h_a, data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, data_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 初始化数据并执行内核
        vector_add<<<32, 256, 0, stream>>>(d_a, d_b, d_c, data_size);
        
        // 释放主机内存
        delete[] h_a;
        delete[] h_b;
    }

    // 测量上下文切换开销
    CUcontext original_ctx;
    cuCtxGetCurrent(&original_ctx);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        CUresult err = cuCtxSetCurrent(contexts[i % NUM_CONTEXTS]);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "第 %d 次切换失败: %d\n", i, err);
            return 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    cuCtxSetCurrent(original_ctx);  // 恢复原始上下文

    // 计算并显示结果
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double avg_ns = duration.count() / (double)ITERATIONS;
    
    printf("总切换次数: %d\n", ITERATIONS);
    printf("总耗时: %.4f ms\n", duration.count() / 1e6);
    printf("平均上下文切换时间: %.4f ns\n", avg_ns);

    // 清理资源
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        if (contexts[i]) {
            cuCtxDestroy(contexts[i]);
        }
    }
    
    return 0;
}
