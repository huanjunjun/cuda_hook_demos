 export LD_PRELOAD=
 cd build
 rm -r ./*
 cmake ..
 make
 export LD_PRELOAD=/root/cuda_hook_demos/cuda_hook_demo1/build/hook_lib/libcuda_hook.so
