cmake_minimum_required(VERSION 3.2.2)
set(OPENMP_MIN_VERSION "3.0")

project(fastLDA C)

set(CMAKE_C_FLAGS "")

find_package(OpenMP	${OPENMP_MIN_VERSION} REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")



