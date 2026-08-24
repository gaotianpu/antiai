cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
# target_link_libraries(example-app "/Users/gaotianpu/Downloads/libtorch")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# /Users/gaotianpu/Downloads/libtorch

# ld: library not found for -lmkl_intel_ilp64
# conda install -c intel mkl==2021.1.1 

# ll /opt/anaconda3/lib | grep intel
