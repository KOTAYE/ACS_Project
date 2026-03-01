cmake_minimum_required(VERSION 3.20)
project(askProjectCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

add_executable(askProjectCUDA
    src/main.cpp
    src/image_io.cpp
    src/tiling.cpp
    src/dct.cpp
    src/quant.cpp
    src/huffman.cpp
    src/metrics.cpp
    src/stb_impl.cpp
    src/codec.cpp
    src/cuda_kernels.cu
)

target_include_directories(askProjectCUDA PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/stb
)

find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(askProjectCUDA PRIVATE OpenMP::OpenMP_CXX)

# Allow .cpp files to include CUDA headers
set_target_properties(askProjectCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
