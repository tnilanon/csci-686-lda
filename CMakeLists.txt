cmake_minimum_required(VERSION 3.2.2)
set(OPENMP_MIN_VERSION 3.0)

project(fast_lda C)

string(ASCII 27 ESCAPE)

# NB: this does not support multi-configurations generator (common in IDE)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "${ESCAPE}[31mNo build type selected, default to Debug${ESCAPE}[0m")
  set(CMAKE_BUILD_TYPE Debug)
else()
  message(STATUS "${ESCAPE}[31mBuild type is ${CMAKE_BUILD_TYPE}${ESCAPE}[0m")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")

find_package(OpenMP	${OPENMP_MIN_VERSION} REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_library(dataset STATIC read_sparse_dataset.c)
add_library(util STATIC util.c)

add_executable(fastLDA scvb0.c)

target_link_libraries(fastLDA dataset util)

