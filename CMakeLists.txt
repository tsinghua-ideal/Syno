# Project settings
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(KAS LANGUAGES C CXX)

# Compiler flags.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Boost
find_package(Boost REQUIRED COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# JSON support
find_package(nlohmann_json REQUIRED)
link_libraries(nlohmann_json::nlohmann_json)

# fmtlib
find_package(fmt REQUIRED)
link_libraries(fmt::fmt-header-only)

# Halide
find_package(Halide REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# GTest
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# OpenMP, which is used to speed up tests. Optional.
find_package(OpenMP)

# Includes and libraries
include_directories(include)

# Sub-directories
add_subdirectory(src)
add_subdirectory(tests)

# Only runtime requires CUDA and Torch.
add_subdirectory(runtime)
