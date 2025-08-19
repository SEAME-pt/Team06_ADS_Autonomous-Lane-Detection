cmake_minimum_required(VERSION 3.10)
project(TrafficSignDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar as bibliotecas externas
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

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
    ${ZeroMQ_INCLUDE_DIRS}
)

# Adicionar o executável
add_executable(objetct_detection
 main.cpp)

# Linkar as bibliotecas
target_link_libraries(objetct_detection
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIB}
    ${NvInferPlugin_LIB}
    ${NvOnnxParser_LIB}
    ${CASADI_LIBRARIES}
    ${ZeroMQ_LIBRARIES}
)

# Configurações adicionais para Jetson
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")

# Mensagens de depuração
message(STATUS "Final CASADI_INCLUDE_DIRS: ${CASADI_INCLUDE_DIRS}")
message(STATUS "Final CASADI_LIBRARIES: ${CASADI_LIBRARIES}")