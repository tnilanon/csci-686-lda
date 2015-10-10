cmake_minimum_required(VERSION 3.2.2)
set(OPENMP_MIN_VERSION 3.0)

project(fast_lda C)

# NB: this does not support multi-configurations generator
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")

find_package(OpenMP	${OPENMP_MIN_VERSION} REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_library(dataset STATIC read_sparse_dataset.c)

add_executable(fastLDA scvb0.c)

target_link_libraries(fastLDA dataset)

