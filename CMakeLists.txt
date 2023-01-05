# Project settings
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(KAS)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Includes and libraries
include_directories(include)

# Halide
find_package(Halide REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# GTest
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Sub-directories
add_subdirectory(src)
add_subdirectory(tests)
