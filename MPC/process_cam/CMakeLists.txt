cmake_minimum_required(VERSION 3.10)
project(LaneDetection)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(lane_detection main.cpp mask_processor.cpp)
target_link_libraries(lane_detection ${OpenCV_LIBS} nvinfer ${CUDA_LIBRARIES})