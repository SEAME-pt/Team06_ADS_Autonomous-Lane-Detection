cmake_minimum_required(VERSION 3.10)
project(LaneDetection)

# Definir o padrão C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar pacotes necessários
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# --- REMOVIDO: JsonCpp configurado manualmente ---
# find_path(JsonCpp_INCLUDE_DIRS
#     NAMES json/json.h
#     PATHS /usr/include /usr/include/jsoncpp
#     NO_CMAKE_FIND_ROOT_PATH
# )
# find_library(JsonCpp_LIBRARIES
#     NAMES jsoncpp
#     PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
#     NO_CMAKE_FIND_ROOT_PATH
# )
# if(NOT JsonCpp_INCLUDE_DIRS)
#     message(FATAL_ERROR "JsonCpp include (json/json.h) nao encontrado. Por favor, instale 'libjsoncpp-dev'.")
# endif()
# if(NOT JsonCpp_LIBRARIES)
#     message(FATAL_ERROR "JsonCpp library (libjsoncpp.so) nao encontrada. Por favor, instale 'libjsoncpp-dev'.")
# endif()
# message(STATUS "JsonCpp configurado manualmente!")
# message(STATUS "JsonCpp_INCLUDE_DIRS: ${JsonCpp_INCLUDE_DIRS}")
# message(STATUS "JsonCpp_LIBRARIES: ${JsonCpp_LIBRARIES}")

# --- Configurar ZeroMQ manualmente (Mantenha esta seção) ---
find_path(ZeroMQ_INCLUDE_DIRS
    NAMES zmq.hpp
    PATHS /usr/local/include /usr/include
    NO_CMAKE_FIND_ROOT_PATH
)
find_library(ZeroMQ_LIBRARIES
    NAMES zmq
    PATHS /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu
    NO_CMAKE_FIND_ROOT_PATH
)
if(NOT ZeroMQ_INCLUDE_DIRS)
    message(FATAL_ERROR "ZeroMQ include (zmq.hpp) nao encontrado. Por favor, instale 'libzmq-dev'.")
endif()
if(NOT ZeroMQ_LIBRARIES)
    message(FATAL_ERROR "ZeroMQ library (libzmq.so) nao encontrada. Por favor, instale 'libzmq-dev'.")
endif()
message(STATUS "ZeroMQ configurado manualmente!")
message(STATUS "ZeroMQ_INCLUDE_DIRS: ${ZeroMQ_INCLUDE_DIRS}")
message(STATUS "ZeroMQ_LIBRARIES: ${ZeroMQ_LIBRARIES}")

# Configurar CasADi manualmente
set(CASADI_INCLUDE_DIRS /usr/local/include)
set(CASADI_LIBRARIES /usr/local/lib/libcasadi.so)
if(NOT EXISTS "${CASADI_INCLUDE_DIRS}/casadi/casadi.hpp")
    message(FATAL_ERROR "CasADi include não encontrado em ${CASADI_INCLUDE_DIRS}/casadi/casadi.hpp")
endif()
if(NOT EXISTS "${CASADI_LIBRARIES}")
    message(FATAL_ERROR "CasADi library não encontrada em ${CASADI_LIBRARIES}")
endif()
message(STATUS "CasADi configurado manualmente!")
message(STATUS "CASADI_INCLUDE_DIRS: ${CASADI_INCLUDE_DIRS}")
message(STATUS "CASADI_LIBRARIES: ${CASADI_LIBRARIES}")

# Definir caminhos para o TensorRT
set(TensorRT_ROOT "/usr/src/tensorrt")
find_library(TensorRT_LIB nvinfer HINTS ${TensorRT_ROOT}/lib)
find_library(NvInferPlugin_LIB nvinfer_plugin HINTS ${TensorRT_ROOT}/lib)
find_library(NvOnnxParser_LIB nvonnxparser HINTS ${TensorRT_ROOT}/lib)

# Incluir diretórios
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
    ${CMAKE_SOURCE_DIR}
    ${CASADI_INCLUDE_DIRS}
    # REMOVIDO: ${JsonCpp_INCLUDE_DIRS}
    ${ZeroMQ_INCLUDE_DIRS}
    ../FServo
    ../BackMotors
    ../Control
)

# Adicionar executável
add_executable(lane_detection
    main.cpp
    lane.cpp
    mask.cpp
    nmpc.cpp
    pid.cpp
    ZmqPublisher.cpp
    ../FServo/FServo.cpp
    ../BackMotors/BackMotors.cpp
    ../Control/ControlAssembly.cpp
    ../Control/Controller.cpp
)

# Linkar bibliotecas
target_link_libraries(lane_detection
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIB}
    ${NvInferPlugin_LIB}
    ${NvOnnxParser_LIB}
    ${CASADI_LIBRARIES}
    # REMOVIDO: ${JsonCpp_LIBRARIES}
    ${ZeroMQ_LIBRARIES}
    rt
)

# Configurações adicionais para Jetson
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")

# Mensagens de depuração
message(STATUS "Final CASADI_INCLUDE_DIRS: ${CASADI_INCLUDE_DIRS}")
message(STATUS "Final CASADI_LIBRARIES: ${CASADI_LIBRARIES}")