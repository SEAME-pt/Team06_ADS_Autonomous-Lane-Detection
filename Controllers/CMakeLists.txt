cmake_minimum_required(VERSION 3.10)

# Project name and version
project(Jetson VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify include directories
include_directories(${PROJECT_SOURCE_DIR}/inc)

# Collect all source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add the executable
add_executable(Controllers ${SOURCES})

