#include <dlfcn.h>
#include <cstdio>
#include <cuda.h>          // 添加CUDA Driver API头文件
#include <cuda_runtime_api.h>

/*
// 示例：
// 假设我们要hook libcudart.so中的cudaMalloc函数
// libcudart.so中的实现（伪代码）
extern "C" cudaError_t cudaMalloc(...) { // 原始实现 }

hook库中的处理：
static auto original = nullptr; // 初始为空
extern "C" cudaMalloc(...) {
    // 在此处通过RTLD_NEXT获取的是libcudart.so中的cudaMalloc函数地址
    if (!original) original = dlsym(RTLD_NEXT, cudaMalloc); // 获取原始地址
    original(...); // 调用原始实现
}
*/

// 原始函数指针声明
static cudaError_t (*original_cudaMalloc)(void** devPtr, size_t size) = nullptr;
static cudaError_t (*original_cudaFree)(void* devPtr) = nullptr;
static cudaError_t (*original_cudaLaunchKernel)(const void* func,
    dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem,
    cudaStream_t stream) = nullptr;
static cudaError_t (*original_cudaLaunch)(const void *func) = nullptr;

// 初始化Driver API
static void initDriverAPI() {
//      使用 static bool initialized 保证只初始化一次
//      避免重复初始化带来的性能损耗
    static bool initialized = false;
    if (!initialized) {
        cuInit(0);
        initialized = true;
    }
}


extern "C" __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func) {
    // 预处理
    printf("[HOOK] cudaLaunch(func=%p)\n", func);
    // 调用原始函数
    if (!original_cudaLaunch) {
        original_cudaLaunch = (decltype(original_cudaLaunch))dlsym(RTLD_NEXT, "cudaLaunch");
    }
    cudaError_t ret = original_cudaLaunch(func);
    // 后处理
    printf("[HOOK] cudaLaunch returned %d\n", ret);
    return ret;
}

// Hook函数实现
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {

    // 预处理
    printf("[HOOK] cudaMalloc(size=%zu)\n", size);


    if (!original_cudaMalloc) {
        original_cudaMalloc = (decltype(original_cudaMalloc))dlsym(RTLD_NEXT, "cudaMalloc");
    }
    cudaError_t ret = original_cudaMalloc(devPtr, size);


    // 后处理
    printf("[HOOK] cudaMalloc returned %d\n", ret);
    return ret;
}



extern "C" cudaError_t cudaFree(void* devPtr) {

    // 预处理
    printf("[HOOK] cudaFree(ptr=%p)\n", devPtr);

    // 调用原始函数
    if (!original_cudaFree) {
        original_cudaFree = (decltype(original_cudaFree))dlsym(RTLD_NEXT, "cudaFree");
    }
    cudaError_t ret = original_cudaFree(devPtr);

    // 后处理
    printf("[HOOK] cudaFree returned %d\n", ret);
    return ret;
}




// 修改函数声明部分
extern "C" __host__ cudaError_t CUDARTAPI cudaLaunchKernel(
    const void* func,
    dim3 gridDim,  dim3 blockDim,
    void** args,
    size_t sharedMem,  cudaStream_t stream)
{
    // 预处理
    printf("[HOOK] cudaLaunchKernel(func=%p, gridDim=%d,%d,%d, blockDim=%d,%d,%d, args=%p, sharedMem=%zu, stream=%p)\n",
           func,
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z,
           args,
           sharedMem,
           stream);

    // 修改符号查找方式
    if (!original_cudaLaunchKernel) {
        original_cudaLaunchKernel = (decltype(original_cudaLaunchKernel))dlsym(
            RTLD_NEXT, 
            "cudaLaunchKernel"
        );
        printf("[HOOK] Original cudaLaunchKernel found at %p\n", original_cudaLaunchKernel);
    }

    // 后处理
    initDriverAPI();
    // 获取内核名称
    char kernelName[256] = "unknown kernel name, please use cuKernelGetName to get";
    // CUresult cuErr = CUDA_SUCCESS;
    // CUfunction f = (CUfunction)func; // 转换函数指针类型
    // cuErr = cuKernelGetName(kernelName, sizeof(kernelName), &f);
    // if (cuErr != CUDA_SUCCESS) {
    //     const char* errStr;
    //     cuGetErrorString(cuErr, &errStr);
    //     printf("[HOOK] Get kernel name failed: %s\n", errStr);
    // }
    printf("[HOOK] cudaLaunchKernel return: %s, Grid(%d,%d,%d), Block(%d,%d,%d)\n",
           kernelName,
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);


    return original_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}