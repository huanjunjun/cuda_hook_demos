#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_SIZE ( 1024 * 1024 * 1024 )  // 1GB
#define NUM_ITER 100

// CUDA错误检查宏
#define checkCudaError(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 测试主机到设备传输
float test_host_to_device(float* h_data, float* d_data, cudaStream_t stream) {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // 预热
    checkCudaError(cudaMemcpyAsync(d_data, h_data, DATA_SIZE, cudaMemcpyHostToDevice, stream));
    
    checkCudaError(cudaEventRecord(start, stream));
    for (int i = 0; i < NUM_ITER; ++i) {
        checkCudaError(cudaMemcpyAsync(d_data, h_data, DATA_SIZE, cudaMemcpyHostToDevice, stream));
    }
    checkCudaError(cudaEventRecord(stop, stream));
    checkCudaError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    
    return milliseconds / NUM_ITER;  // 返回平均时间
}

// 测试设备到主机传输
float test_device_to_host(float* h_data, float* d_data, cudaStream_t stream) {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // 预热
    checkCudaError(cudaMemcpyAsync(h_data, d_data, DATA_SIZE, cudaMemcpyDeviceToHost, stream));
    
    checkCudaError(cudaEventRecord(start, stream));
    for (int i = 0; i < NUM_ITER; ++i) {
        checkCudaError(cudaMemcpyAsync(h_data, d_data, DATA_SIZE, cudaMemcpyDeviceToHost, stream));
    }
    checkCudaError(cudaEventRecord(stop, stream));
    checkCudaError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    
    return milliseconds / NUM_ITER;  // 返回平均时间
}

// 新增分页内存传输测试函数
float test_pageable_transfer(void* src, void* dst, cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // 同步预热
    checkCudaError(cudaMemcpy(dst, src, DATA_SIZE, kind));
    
    checkCudaError(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITER; ++i) {
        checkCudaError(cudaMemcpy(dst, src, DATA_SIZE, kind));
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    
    return milliseconds / NUM_ITER;
}

int main() {
    float *h_data, *d_data, *h_pageable;
    cudaStream_t stream;
    
    // 分配pinned host内存
    checkCudaError(cudaMallocHost(&h_data, DATA_SIZE));
    // 分配分页host内存
    h_pageable = (float*)malloc(DATA_SIZE);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        exit(EXIT_FAILURE);
    }
    // 分配device内存
    checkCudaError(cudaMalloc(&d_data, DATA_SIZE));
    // 创建CUDA流
    checkCudaError(cudaStreamCreate(&stream));

    // 测试Host->Device
    float h2d_time = test_host_to_device(h_data, d_data, stream);
    printf("Host->Device 传输时间: %.3f ms\n带宽: %.2f GB/s\n", 
           h2d_time, 
           (DATA_SIZE / (1024.0f * 1024.0f * 1024.0f)) / (h2d_time / 1000.0f));

    // 测试Device->Host
    float d2h_time = test_device_to_host(h_data, d_data, stream);
    printf("\nDevice->Host 传输时间: %.3f ms\n带宽: %.2f GB/s\n", 
           d2h_time, 
           (DATA_SIZE / (1024.0f * 1024.0f * 1024.0f)) / (d2h_time / 1000.0f));

    // 新增分页内存测试
    float h2d_page_time = test_pageable_transfer(h_pageable, d_data, cudaMemcpyHostToDevice);
    printf("\nPageable->Device 传输时间: %.3f ms\n带宽: %.2f GB/s",
           h2d_page_time, 
           (DATA_SIZE / (1024.0f * 1024.0f * 1024.0f)) / (h2d_page_time / 1000.0f));

    float d2h_page_time = test_pageable_transfer(d_data, h_pageable, cudaMemcpyDeviceToHost);
    printf("\n\nDevice->Pageable 传输时间: %.3f ms\n带宽: %.2f GB/s",
           d2h_page_time,
           (DATA_SIZE / (1024.0f * 1024.0f * 1024.0f)) / (d2h_page_time / 1000.0f));

    // 新增分页内存释放
    free(h_pageable);
    // 清理资源
    checkCudaError(cudaFreeHost(h_data));
    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaStreamDestroy(stream));

    return 0;
}
