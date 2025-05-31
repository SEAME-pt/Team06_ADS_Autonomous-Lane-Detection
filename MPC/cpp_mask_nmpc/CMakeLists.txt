cmake_minimum_required(VERSION 3.10)
project(NMPC_Lane_Following)

find_package(OpenCV REQUIRED)
find_package(Casadi REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CASADI_INCLUDE_DIR})

add_executable(nmpc_lane_following
    main.cpp
    nmpc_controller.cpp
    lane_processor.cpp
    vehicle_interface.cpp
)

target_link_libraries(nmpc_lane_following ${OpenCV_LIBRARIES} casadi)