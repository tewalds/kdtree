cmake_minimum_required(VERSION 3.14)
project(kdtree VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(kdtree INTERFACE)
target_include_directories(kdtree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Tests
add_executable(kdtree_test kdtree_test.cc catch2/catch_amalgamated.cpp)
target_link_libraries(kdtree_test PRIVATE kdtree)
target_include_directories(kdtree_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kdtree_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

enable_testing()
add_test(NAME kdtree_test COMMAND kdtree_test '--skip-benchmarks')
add_test(NAME kdtree_benchmark COMMAND kdtree_test '[benchmark]')

# Installation
install(FILES kdtree.h DESTINATION include)
