cmake_minimum_required(VERSION 3.20)
project(flipbook_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(ROOT ${CMAKE_SOURCE_DIR}/..)

set(SHARED_SRC
    ${ROOT}/src/image_io.cpp
    ${ROOT}/src/tiling.cpp
    ${ROOT}/src/dct.cpp
    ${ROOT}/src/quant.cpp
    ${ROOT}/src/huffman.cpp
    ${ROOT}/src/metrics.cpp
    ${ROOT}/src/stb_impl.cpp
)

set(INC_DIRS ${ROOT}/src ${ROOT}/stb)

# ── CUDA ──────────────────────────────────────────────────
add_executable(bench_cuda
    main_bench.cpp
    ${ROOT}/src/codec.cpp
    ${ROOT}/src/cuda_kernels.cu
    ${SHARED_SRC}
)
target_include_directories(bench_cuda PRIVATE ${INC_DIRS})
target_compile_definitions(bench_cuda PRIVATE USE_CUDA)
set_target_properties(bench_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ── OpenMP ────────────────────────────────────────────────
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(bench_omp
    main_bench.cpp
    codec_cpu.cpp
    ${SHARED_SRC}
)
target_include_directories(bench_omp PRIVATE ${INC_DIRS})
target_compile_definitions(bench_omp PRIVATE USE_OMP)
target_link_libraries(bench_omp PRIVATE OpenMP::OpenMP_CXX)

# ── Serial (single-threaded) ─────────────────────────────
add_executable(bench_serial
    main_bench.cpp
    codec_cpu.cpp
    ${SHARED_SRC}
)
target_include_directories(bench_serial PRIVATE ${INC_DIRS})
