# Project settings
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(KAS LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Includes and libraries
include_directories(include)

# Halide
find_package(Halide REQUIRED)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# GTest
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Sub-directories
add_subdirectory(src)
add_subdirectory(tests)
