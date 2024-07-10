cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Experimental")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

project(Specular)

# Get the absolute path to the project root directory
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}" ABSOLUTE)

# Pass the project root directory as a preprocessor definition
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
add_definitions(-DPOJECT_BUILD_DIR="${PROJECT_ROOT_DIR}/build")

find_package(Eigen3 REQUIRED)

include_directories(PUBLIC include ${EIGEN3_INCLUDE_DIR})

add_executable(specular_example examples/specular_example.cpp)
add_executable(specular_example_fast examples/specular_example_fast.cpp)