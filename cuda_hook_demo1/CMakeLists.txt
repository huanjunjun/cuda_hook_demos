cmake_minimum_required(VERSION 3.10)
project(cuda_hook_demo)

enable_language(CUDA)
# 这一行是必须的，解决一个问题
# CMake Error: Cannot determine link language for target "cuda_hook".
# CMake Error: CMake can not determine linker language for target: cuda_hook
# CMake Error: Cannot determine link language for target "test_app".
# CMake Error: CMake can not determine linker language for target: test_app

add_subdirectory(hook_lib)
add_subdirectory(test_app)