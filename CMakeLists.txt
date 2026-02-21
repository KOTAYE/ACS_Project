cmake_minimum_required(VERSION 3.20)
project(askProjectCUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(askProjectCUDA
    src/main.cpp
    src/image_io.cpp
    src/tiling.cpp
    src/dct.cpp
    src/quant.cpp
    src/huffman.cpp
    src/metrics.cpp
    src/stb_impl.cpp
)

target_include_directories(askProjectCUDA PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/stb
)
