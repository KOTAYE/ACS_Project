cmake_minimum_required(VERSION 3.20)
project(askProjectCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

set(SHARED_SRC
    src/image_io.cpp
    src/tiling.cpp
    src/dct.cpp
    src/quant.cpp
    src/huffman.cpp
    src/metrics.cpp
    src/stb_impl.cpp
)

set(INC_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/stb
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/external
)

find_package(ZLIB REQUIRED)

add_library(tinyexr_c STATIC src/tinyexr_impl.cpp)
target_include_directories(tinyexr_c PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/external
)
target_link_libraries(tinyexr_c PRIVATE ZLIB::ZLIB)

add_executable(flipbook_cuda
    src/main_cuda.cpp
    src/codec.cpp
    src/realtime/in_memory_codec.cpp
    src/cuda_kernels.cu
    src/rle_gpu.cu
    ${SHARED_SRC}
)
target_include_directories(flipbook_cuda PRIVATE ${INC_DIRS})
target_link_libraries(flipbook_cuda PRIVATE tinyexr_c)
set_target_properties(flipbook_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(realtime_codec SHARED
    src/realtime/in_memory_codec.cpp
    src/realtime/in_memory_codec_c_api.cpp
    src/cuda_kernels.cu
    src/rle_gpu.cu
    ${SHARED_SRC}
)
target_include_directories(realtime_codec PRIVATE ${INC_DIRS})
target_link_libraries(realtime_codec PRIVATE tinyexr_c)
set_target_properties(realtime_codec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(flipbook_omp
    src/main_omp.cpp
    bench/codec_cpu.cpp
    ${SHARED_SRC}
)
add_library(tinyexr STATIC
    external/tinyexr.cc
)
target_include_directories(tinyexr PUBLIC
    ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(flipbook_omp PRIVATE tinyexr)
target_include_directories(flipbook_omp PRIVATE ${INC_DIRS})
target_compile_definitions(flipbook_omp PRIVATE USE_OMP)
target_link_libraries(flipbook_omp PRIVATE OpenMP::OpenMP_CXX tinyexr_c)

add_executable(flipbook_serial
    src/main_serial.cpp
    bench/codec_cpu.cpp
    ${SHARED_SRC}
)
target_include_directories(flipbook_serial PRIVATE ${INC_DIRS})
target_link_libraries(flipbook_serial PRIVATE tinyexr_c)

add_executable(entropy_reference
    bench/entropy_reference.cpp
    src/arithmetic_ref.cpp
    ${SHARED_SRC}
)
target_include_directories(entropy_reference PRIVATE ${INC_DIRS})
target_link_libraries(entropy_reference PRIVATE tinyexr_c)

enable_testing()
add_test(NAME entropy_reference_selftest COMMAND $<TARGET_FILE:entropy_reference> --self-test)