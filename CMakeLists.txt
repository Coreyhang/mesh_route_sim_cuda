cmake_minimum_required(VERSION 3.5)

project (cuda)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

find_package(CUDA)
if (CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(DotProduct src/route_sim.cu)
else()
    message("CUDA not found!")
endif()