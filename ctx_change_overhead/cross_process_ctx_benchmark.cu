#include <sys/mman.h>
#include <fcntl.h>
#include <cuda.h>
#include <unistd.h>
#include <sys/types.h>
#include <chrono>
#include <stdio.h>
#include <atomic>

// 共享内存结构
struct SyncData {
    std::atomic<int> ready_flag;     // 就绪信号
    std::atomic<int> timing_data[4]; // [0]启动时间低32位, [1]启动时间高32位, [2]结束时间低32位, [3]结束时间高32位
};

__global__ void workload_kernel() {
    // 占用计算资源的内核
    for(int i=0; i<1000; i++) {
        __sinf(1.0f);
    }
}

void worker_process(int proc_id) {
    // 创建CUDA上下文
    // CUcontext ctx;
    // cuCtxCreate(&ctx, 0, 0);

    // 映射共享内存
    int fd = shm_open("/cuda_ctx_bench", O_RDWR, 0666);
    SyncData* shared = (SyncData*)mmap(0, sizeof(SyncData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

    while(true) {
        // 等待同步信号
        while(shared->ready_flag.load() != proc_id);

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        uint64_t start_ns = start.time_since_epoch().count();
        shared->timing_data[0].store(static_cast<int>(start_ns & 0xFFFFFFFF));
        shared->timing_data[1].store(static_cast<int>(start_ns >> 32));

        // 执行上下文切换和计算，通过触发核函数触发上下文切换
        // cuCtxSetCurrent(ctx);
        workload_kernel<<<32, 256>>>();
        cudaDeviceSynchronize();

        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();
        uint64_t end_ns = end.time_since_epoch().count();
        shared->timing_data[2].store(static_cast<int>(end_ns & 0xFFFFFFFF));
        shared->timing_data[3].store(static_cast<int>(end_ns >> 32));

        // 通知控制进程
        shared->ready_flag.store(0xFFFF);
    }
}

int main(int argc, char** argv) {
    
    // 初始化共享内存
    int fd = shm_open("/cuda_ctx_bench", O_CREAT|O_RDWR, 0666);
    ftruncate(fd, sizeof(SyncData));
    SyncData* shared = (SyncData*)mmap(0, sizeof(SyncData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    shared->ready_flag.store(0);

    // 启动工作进程
    int process_num = 10;
    for(int i=0; i<process_num; i++) {
        if(fork() == 0) {
            worker_process(i+1);
            exit(0);
        }
    }

    uint64_t total_time = 0;

    // 修改后的主控测试循环
    int iteation = 1000000;
    for(int test=0; test<iteation; test++) {
        // 依次触发4个进程，无内层循环
        shared->ready_flag.store((test % process_num) + 1);
        
        // 添加超时检测
        auto timeout = std::chrono::high_resolution_clock::now() + std::chrono::seconds(5);
        while(shared->ready_flag.load() != 0xFFFF) {
            if(std::chrono::high_resolution_clock::now() > timeout) {
                printf("超时! 测试中断在迭代 %d\n", test);
                exit(1);
            }
        }

        // 获取时间差
        uint64_t start = (uint64_t(shared->timing_data[1].load()) << 32) | shared->timing_data[0].load();
        uint64_t end = (uint64_t(shared->timing_data[3].load()) << 32) | shared->timing_data[2].load();
        printf("进程%d切换耗时: %.2f us\n", (test % process_num)+1, (end - start)/1000.0);
        if(end - start > 10000000000) {
            printf("发现异常：进程%d切换耗时超过10s\n", (test % process_num)+1);
            // exit(1);
        }else{
            total_time += (end - start);
        }
        
        
    }

    printf("总耗时: %.2f us\n", total_time/1000.0);
    printf("平均耗时: %.2f us\n", total_time/(1000.0*iteation));
    
    return 0;
}
