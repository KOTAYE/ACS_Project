cmake_minimum_required(VERSION 3.20)
project(askProjectCUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(askProjectCUDA
    main.cpp
    image_io.cpp
    tiling.cpp
    dct.cpp
    metrics.cpp
    stb_impl.cpp
)

target_include_directories(askProjectCUDA PRIVATE ${CMAKE_SOURCE_DIR}/stb)
