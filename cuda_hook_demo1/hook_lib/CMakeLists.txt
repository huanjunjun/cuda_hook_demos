# 添加生成的动态库
add_library(cuda_hook SHARED cuda_hook.cu)

# 添加头文件路径
target_link_libraries(cuda_hook PRIVATE cudart cuda)

# 添加编译选项
set_target_properties(cuda_hook PROPERTIES POSITION_INDEPENDENT_CODE ON)

