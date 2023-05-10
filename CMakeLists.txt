# Project settings
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(KAS LANGUAGES C CXX)

# Compiler flags.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)

# We first find libTorch. This is because PyTorch requires nvcc to support the same C++ standard as the host compiler, which may be too demanding.
# CUDA
find_package(CUDAToolkit REQUIRED)
# PyTorch
find_package(Torch REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Boost
find_package(Boost REQUIRED COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# fmtlib
find_package(fmt REQUIRED)
link_libraries(fmt::fmt)

# Halide
find_package(Halide REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# GTest
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Includes and libraries
include_directories(include)

# Sub-directories
add_subdirectory(src)
add_subdirectory(tests)
