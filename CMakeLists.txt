cmake_minimum_required(VERSION 3.20)
project(askProjectCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

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
)

# ── CUDA target ──────────────────────────────────────────────
add_executable(flipbook_cuda
    src/main_cuda.cpp
    src/codec.cpp
    src/cuda_kernels.cu
    ${SHARED_SRC}
)
target_include_directories(flipbook_cuda PRIVATE ${INC_DIRS})
set_target_properties(flipbook_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ── OpenMP target ────────────────────────────────────────────
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(flipbook_omp
    src/main_omp.cpp
    bench/codec_cpu.cpp
    ${SHARED_SRC}
)
target_include_directories(flipbook_omp PRIVATE ${INC_DIRS})
target_compile_definitions(flipbook_omp PRIVATE USE_OMP)
target_link_libraries(flipbook_omp PRIVATE OpenMP::OpenMP_CXX)

# ── Serial target (single-threaded) ─────────────────────────
add_executable(flipbook_serial
    src/main_serial.cpp
    bench/codec_cpu.cpp
    ${SHARED_SRC}
)
target_include_directories(flipbook_serial PRIVATE ${INC_DIRS})
