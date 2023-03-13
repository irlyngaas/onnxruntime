#!/bin/bash

set -ex 

build_dir="build"
config="Release"

rocm_home="/opt/rocm-5.4.0"

./build.sh --update \
    --build_dir ${build_dir} \
    --config ${config} \
    --cmake_extra_defines \
        CMAKE_HIP_COMPILER=/opt/rocm-5.4.0/llvm/bin/clang++ \
        onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        onnxruntime_ENABLE_ATEN=OFF \
    --skip_submodule_sync --skip_tests \
    --use_rocm --rocm_home=${rocm_home} --nccl_home=${rocm_home} \
    --build_wheel

cmake --build ${build_dir}/${config} --target kernel_explorer --parallel
#cmake --build ${build_dir}/${config} --target kernel_explorer
        #onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        #onnxruntime_ENABLE_PYTHON=ON \
        #onnxruntime_PREBUILT_PYTORCH_PATH=/gpfs/alpine/med106/world-shared/irl1/ckIntegration/conda540/lib/python3.9/site-packages \
