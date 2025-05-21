// gpu_worker.cu
#include <cstdio>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

__global__ void busy_kernel(float* data, int size) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < size) {
        float val = data[idx];
        val = sqrtf(powf(val, 3.2f) + 1.0f);
        data[idx] = val;
    }
}

int main(int argc, char** argv) {
    const int DATA_SIZE = 1024*1024*128;  // 512MB数据
    float *d_data;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // GPU资源分配
    cudaMalloc(&d_data, DATA_SIZE*sizeof(float));
    
    // 执行计算密集型任务
    dim3 block(256);
    dim3 grid((DATA_SIZE + block.x -1)/block.x);
    
    uint64_t c=0;
    for(int i=0; i<1000; ++i) {  // 重复执行1000次
        busy_kernel<<<grid, block>>>(d_data, DATA_SIZE);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    cudaDeviceSynchronize();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    printf("Process %s execution time: %.2fms\n", 
        argv[0],
        std::chrono::duration<double, std::milli>(t_end-t_start).count());
    
    cudaFree(d_data);
    return 0;
}

// ./worker > log1.txt&
// ./worker > log2.txt&
